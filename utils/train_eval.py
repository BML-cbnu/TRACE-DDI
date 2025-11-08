# utils/train_eval.py
import os
import json
import logging
import torch
import matplotlib.pyplot as plt

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report

from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from .data import DDIDataset, custom_collate_fn
from .model import DDIClassifier

logger = logging.getLogger(__name__)

# --------- Plots (label distribution) ---------
def _plot_label_pie(train_df, code_to_label, log_dir, fold):
    counts = train_df['interaction'].value_counts()
    top5 = counts.nlargest(5)
    others = counts.sum() - top5.sum()
    labels = [code_to_label.get(i, str(i)) for i in top5.index] + ['Other']
    sizes = top5.values.tolist() + [others]

    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title(f'Fold {fold} Label Distribution (Top 5 + Other)')
    os.makedirs(log_dir, exist_ok=True)
    out_path = os.path.join(log_dir, f'fold_{fold}_label_distribution.png')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    logger.info(f"Saved label distribution pie chart to {out_path}")

# --------- Eval & save ---------
def _eval_save(model: DDIClassifier, val_loader, device, args, fold: int):
    enc_ckpt = os.path.join(args.model_save_dir, f"best_encoder_fold{fold}.pth")
    mod_ckpt = os.path.join(args.model_save_dir, f"best_model_fold{fold}.pth")
    res_file = os.path.join(args.result_dir,   f"results_fold{fold}.txt")
    os.makedirs(args.model_save_dir, exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)

    model.to(device).eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for t1,t2,v1,v2,y,a1,a2 in val_loader:
            t1 = t1.to(device); t2 = t2.to(device)
            v1 = v1.to(device); v2 = v2.to(device)
            y  = y.to(device)
            a1 = a1.to(device); a2 = a2.to(device)
            logits = model(t1,t2,v1,v2,a1,a2)
            preds  = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.cpu().tolist())

    acc = float(accuracy_score(all_labels, all_preds))
    rep = classification_report(all_labels, all_preds, output_dict=True, digits=4, zero_division=1)
    weighted = rep['weighted avg']
    summary = {
        "accuracy":  acc,
        "precision": float(weighted['precision']),
        "recall":    float(weighted['recall']),
        "f1":        float(weighted['f1-score'])
    }
    logger.info(f"[Fold {fold}] metrics: {json.dumps(summary, ensure_ascii=False)}")

    with open(res_file, 'w') as f:
        f.write(f"Accuracy:  {summary['accuracy']:.4f}\n")
        f.write(f"Precision: {summary['precision']:.4f}\n")
        f.write(f"Recall:    {summary['recall']:.4f}\n")
        f.write(f"F1-score:  {summary['f1']:.4f}\n\n")
        f.write(classification_report(all_labels, all_preds, digits=4, zero_division=1))

    # Save encoder & full model weights
    torch.save(model.smiles_enc.state_dict(), enc_ckpt)
    torch.save(model.state_dict(),           mod_ckpt)
    logger.info(f"Saved encoder to {enc_ckpt}")
    logger.info(f"Saved model   to {mod_ckpt}")

# --------- CV runner ---------
def run_stratified_cv(ddi_df, tokenizer, code_to_label,
                      compound_vectors, drug_to_vector_index, args):
    """Run K-fold with logs at the exact moments real work happens."""
    import pynvml
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vocab_size    = len(tokenizer.vocab)
    graph_vec_dim = compound_vectors.shape[1]
    num_classes   = ddi_df['interaction'].nunique()

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42)
    pynvml.nvmlInit()
    try:
        for fold, (tr_idx, va_idx) in enumerate(skf.split(ddi_df, ddi_df['interaction']), start=1):
            logger.info(f"=== Fold {fold} Training ===")

            train_df, val_df = ddi_df.iloc[tr_idx], ddi_df.iloc[va_idx]
            _plot_label_pie(train_df, code_to_label, args.log_dir, fold)

            train_ds = DDIDataset(train_df, tokenizer, compound_vectors, drug_to_vector_index)
            val_ds   = DDIDataset(val_df,   tokenizer, compound_vectors, drug_to_vector_index)

            train_loader = DataLoader(
                train_ds, batch_size=args.batch_size, shuffle=True,
                collate_fn=custom_collate_fn, num_workers=4
            )
            val_loader = DataLoader(
                val_ds, batch_size=args.batch_size, shuffle=False,
                collate_fn=custom_collate_fn, num_workers=2
            )

            model = DDIClassifier(
                vocab=vocab_size,
                emb_dim=args.embedding_dim,
                d_model=args.d_model,
                graph_dim=graph_vec_dim,
                hid=args.hidden_dim,
                num_cls=num_classes,
                clf_drop=args.classifier_dropout,
                gat_drop=args.gat_dropout,
                lr=args.lr,
                nhead=args.nhead,
                num_layers=args.num_encoder_layers,
                ff=args.dim_feedforward,
                gat_alpha=args.gat_alpha
            )
            logger.info(f"Model structure:\n{model}")

            csv_logger     = CSVLogger(args.log_dir, name=f"fold_{fold}")
            early_stopping = EarlyStopping(monitor='val_acc', patience=10, mode='max')
            checkpoint_cb  = ModelCheckpoint(
                dirpath=args.model_save_dir,
                filename=f"fold{fold}" + "-{epoch:02d}-{val_acc:.4f}",
                monitor="val_acc", mode="max", save_top_k=1
            )

            # Create trainer AFTER logs so ordering matches original behavior
            trainer = pl.Trainer(
                max_epochs=args.num_epochs,
                logger=csv_logger,
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                devices=1 if torch.cuda.is_available() else None,
                callbacks=[early_stopping, checkpoint_cb],
                enable_progress_bar=False
            )

            # Fit
            trainer.fit(model, train_loader, val_loader)

            # Load best checkpoint if any
            best_path = checkpoint_cb.best_model_path
            if best_path and os.path.exists(best_path):
                logger.info(f"Loading best checkpoint: {best_path}")
                best_model = DDIClassifier(
                    vocab=vocab_size,
                    emb_dim=args.embedding_dim,
                    d_model=args.d_model,
                    graph_dim=graph_vec_dim,
                    hid=args.hidden_dim,
                    num_cls=num_classes,
                    clf_drop=args.classifier_dropout,
                    gat_drop=args.gat_dropout,
                    lr=args.lr,
                    nhead=args.nhead,
                    num_layers=args.num_encoder_layers,
                    ff=args.dim_feedforward,
                    gat_alpha=args.gat_alpha
                )
                ckpt = torch.load(best_path, map_location=device)
                if "state_dict" in ckpt:
                    best_model.load_state_dict(ckpt["state_dict"])
                else:
                    best_model.load_state_dict(ckpt)
                model = best_model
            else:
                logger.warning("Best checkpoint not found; using the last trained weights.")

            # Evaluate + save
            _eval_save(model, val_loader, device, args, fold)
    finally:
        pynvml.nvmlShutdown()