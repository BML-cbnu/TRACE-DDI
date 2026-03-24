from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, cast

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from utils.common import ensure_dirs, init_nvml_if_available, set_seeds, shutdown_nvml
from utils.dataset import DDIDataset, custom_collate_fn
from utils.metrics import (
    classification_report_dict,
    classification_report_str,
    compute_binary_metrics,
)
from utils.model import DDIClassifier
from utils.preprocess import (
    canonicalize_df_pairs,
    compute_drug_overlap_stats,
    dedup_pairs,
    encode_multiclass_labels,
    make_drug_holdout_split_strict,
)
from utils.tokenizer import SMILESTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TRACE-DDI")


@dataclass(frozen=True)
class TrainArgs:
    ddi_path: str
    smiles_path: str
    vec_path: str
    log_dir: str
    model_dir: str
    result_dir: str

    eval_mode: str
    pair_dedup: bool
    build_vocab_from_train: bool

    holdout_frac: float

    num_epochs: int
    batch_size: int
    lr: float
    seed: int
    max_len: int
    n_splits: int

    d_model: int
    embedding_dim: int
    nhead: int
    num_encoder_layers: int
    dim_feedforward: int

    hidden_dim: int
    classifier_dropout: float

    gat_dropout: float
    gat_alpha: float
    num_heads: int

    case_sensitive: bool
    num_workers_train: int
    num_workers_val: int


def _load_state_dict_safely(model: nn.Module, path: str, device: torch.device) -> None:
    ckpt_any: Any = torch.load(path, map_location=device)
    if isinstance(ckpt_any, dict) and "state_dict" in ckpt_any:
        state_any: Any = ckpt_any["state_dict"]
    else:
        state_any = ckpt_any
    model.load_state_dict(cast(Dict[str, torch.Tensor], state_any), strict=False)


def _save_encoder_and_model(
    model: DDIClassifier,
    model_dir: str,
    tag: str,
) -> Tuple[str, str]:
    # Save encoder and full model
    os.makedirs(model_dir, exist_ok=True)

    enc_path = os.path.join(model_dir, f"best_encoder_{tag}.pth")
    model_path = os.path.join(model_dir, f"best_model_{tag}.pth")

    torch.save(model.smiles_enc.state_dict(), enc_path)
    torch.save(model.state_dict(), model_path)

    logger.info("Saved encoder to %s", enc_path)
    logger.info("Saved model to %s", model_path)

    return enc_path, model_path


def train_main(a: TrainArgs) -> None:
    ensure_dirs(a.log_dir, a.model_dir, a.result_dir)
    set_seeds(a.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    ddi_df = pd.read_csv(
        a.ddi_path,
        sep="\t",
        header=None,
        names=["drug1", "drug2", "interaction"],
    )
    ddi_df["drug1"] = ddi_df["drug1"].astype(str)
    ddi_df["drug2"] = ddi_df["drug2"].astype(str)

    if a.pair_dedup:
        ddi_df = canonicalize_df_pairs(ddi_df)
        ddi_df = dedup_pairs(ddi_df, ["drug1", "drug2", "interaction"])

    smiles_df = pd.read_csv(
        a.smiles_path,
        sep="\t",
        header=None,
        names=["drug", "smiles"],
    )
    smiles_df["drug"] = smiles_df["drug"].astype(str)

    ddi_df = (
        ddi_df.merge(smiles_df, left_on="drug1", right_on="drug")
        .rename(columns={"smiles": "smiles_x"})
        .drop("drug", axis=1)
    )
    ddi_df = (
        ddi_df.merge(smiles_df, left_on="drug2", right_on="drug")
        .rename(columns={"smiles": "smiles_y"})
        .drop("drug", axis=1)
    )

    vec_df = pd.read_csv(a.vec_path, index_col=0)
    comp_vecs = vec_df.to_numpy()
    d2i: Dict[str, int] = {
        str(drug): i for i, drug in enumerate(vec_df.index.astype(str).tolist())
    }
    graph_dim = int(comp_vecs.shape[1])

    d2i_keys_list: List[str] = list(d2i.keys())
    before = len(ddi_df)
    mask = ddi_df["drug1"].isin(d2i_keys_list) & ddi_df["drug2"].isin(d2i_keys_list)
    ddi_df = ddi_df.loc[mask].reset_index(drop=True)
    ddi_df = cast(pd.DataFrame, ddi_df)
    after = len(ddi_df)

    if after < before:
        logger.info("Filtered pairs without vectors: %d removed.", before - after)

    nvml_enabled = init_nvml_if_available()
    try:
        if a.eval_mode == "multiclass_cv":
            ddi_df, num_classes = encode_multiclass_labels(ddi_df, col="interaction")

            skf = StratifiedKFold(
                n_splits=a.n_splits,
                shuffle=True,
                random_state=a.seed,
            )

            for fold, (tr_idx, va_idx) in enumerate(
                skf.split(ddi_df, ddi_df["interaction"]),
                start=1,
            ):
                train_df = ddi_df.iloc[tr_idx].copy()
                val_df = ddi_df.iloc[va_idx].copy()

                stats = compute_drug_overlap_stats(train_df, val_df)
                logger.info(
                    "[multiclass_cv] fold=%d train_drugs=%d val_drugs=%d shared_drugs=%d",
                    fold,
                    stats["n_train_drugs"],
                    stats["n_val_drugs"],
                    stats["n_shared_drugs"],
                )

                overlap_file = os.path.join(
                    a.result_dir, f"multiclass_drug_overlap_fold{fold}.txt"
                )
                with open(overlap_file, "w", encoding="utf-8") as f:
                    f.write(f"Fold: {fold}\n")
                    f.write(f"Train drugs: {stats['n_train_drugs']}\n")
                    f.write(f"Val drugs: {stats['n_val_drugs']}\n")
                    f.write(f"Shared drugs (train ∩ val): {stats['n_shared_drugs']}\n")

                if a.build_vocab_from_train:
                    smiles_train = (
                        pd.concat([train_df["smiles_x"], train_df["smiles_y"]])
                        .astype(str)
                        .unique()
                        .tolist()
                    )
                    tokenizer = SMILESTokenizer(
                        smiles_train,
                        case_sensitive=a.case_sensitive,
                    )
                else:
                    all_smiles = (
                        pd.concat([ddi_df["smiles_x"], ddi_df["smiles_y"]])
                        .astype(str)
                        .unique()
                        .tolist()
                    )
                    tokenizer = SMILESTokenizer(
                        all_smiles,
                        case_sensitive=a.case_sensitive,
                    )

                train_ds = DDIDataset(train_df, tokenizer, comp_vecs, d2i, a.max_len)
                val_ds = DDIDataset(val_df, tokenizer, comp_vecs, d2i, a.max_len)

                train_loader = DataLoader(
                    train_ds,
                    batch_size=a.batch_size,
                    shuffle=True,
                    collate_fn=custom_collate_fn,
                    num_workers=a.num_workers_train,
                )
                val_loader = DataLoader(
                    val_ds,
                    batch_size=a.batch_size,
                    shuffle=False,
                    collate_fn=custom_collate_fn,
                    num_workers=a.num_workers_val,
                )

                model = DDIClassifier(
                    vocab_size=len(tokenizer.vocab),
                    emb_dim=a.embedding_dim,
                    d_model=a.d_model,
                    graph_dim=graph_dim,
                    hid_dim=a.hidden_dim,
                    num_classes=num_classes,
                    clf_drop=a.classifier_dropout,
                    gat_drop=a.gat_dropout,
                    lr=a.lr,
                    tf_nhead=a.nhead,
                    tf_layers=a.num_encoder_layers,
                    tf_dim_ff=a.dim_feedforward,
                    gat_heads=a.num_heads,
                    gat_alpha=a.gat_alpha,
                )

                csv_logger = CSVLogger(a.log_dir, name=f"fold_{fold}")
                early_stopping = EarlyStopping(
                    monitor="val_acc",
                    patience=10,
                    mode="max",
                )
                checkpoint_cb = ModelCheckpoint(
                    dirpath=a.model_dir,
                    filename=f"fold{fold}" + "-{epoch:02d}-{val_acc:.4f}",
                    monitor="val_acc",
                    mode="max",
                    save_top_k=1,
                )

                trainer = pl.Trainer(
                    max_epochs=a.num_epochs,
                    logger=csv_logger,
                    accelerator="gpu" if torch.cuda.is_available() else "cpu",
                    devices=1,
                    callbacks=[early_stopping, checkpoint_cb],
                    enable_progress_bar=False,
                )
                trainer.fit(model, train_loader, val_loader)

                best_path = checkpoint_cb.best_model_path
                if best_path and os.path.exists(best_path):
                    _load_state_dict_safely(model, best_path, device)
                    _save_encoder_and_model(model, a.model_dir, f"fold{fold}")
                else:
                    logger.warning(
                        "[multiclass_cv] best checkpoint not found for fold %d; saving current model weights.",
                        fold,
                    )
                    _save_encoder_and_model(model, a.model_dir, f"fold{fold}")

                model.to(device).eval()
                all_labels: List[int] = []
                all_preds: List[int] = []

                with torch.no_grad():
                    for t1, t2, v1, v2, y, a1_, a2_ in val_loader:
                        logits = model(
                            t1.to(device),
                            t2.to(device),
                            v1.to(device),
                            v2.to(device),
                            a1_.to(device),
                            a2_.to(device),
                        )
                        preds = torch.argmax(logits, dim=1)
                        all_preds.extend(preds.detach().cpu().tolist())
                        all_labels.extend(y.detach().cpu().tolist())

                acc = float(accuracy_score(all_labels, all_preds))
                rep = classification_report_dict(
                    all_labels,
                    all_preds,
                    digits=4,
                    zero_division=1,
                )
                weighted_any = rep.get("weighted avg")
                if not isinstance(weighted_any, dict):
                    raise RuntimeError("classification_report missing 'weighted avg'")
                weighted = cast(Dict[str, Any], weighted_any)

                f1 = float(weighted["f1-score"])
                precision = float(weighted["precision"])
                recall = float(weighted["recall"])
                rep_str = classification_report_str(
                    all_labels,
                    all_preds,
                    digits=4,
                    zero_division=1,
                )

                out_file = os.path.join(
                    a.result_dir,
                    f"multiclass_results_fold{fold}.txt",
                )
                with open(out_file, "w", encoding="utf-8") as f:
                    f.write(f"Accuracy:  {acc:.4f}\n")
                    f.write(f"Precision: {precision:.4f}\n")
                    f.write(f"Recall:    {recall:.4f}\n")
                    f.write(f"F1-score:  {f1:.4f}\n\n")
                    f.write(rep_str)

                logger.info("[multiclass_cv] fold=%d acc=%.4f f1=%.4f", fold, acc, f1)

        else:
            uniq_vals = pd.Series(ddi_df["interaction"]).astype(int).unique()
            uniq_set: set[int] = {int(x) for x in uniq_vals}
            if uniq_set - {0, 1}:
                raise ValueError(
                    f"[drug_holdout_binary] expected binary labels in interaction; got {sorted(uniq_set)}"
                )

            ddi_df_holdout = cast(pd.DataFrame, ddi_df)
            train_df, test_df, holdout_drugs, dropped = make_drug_holdout_split_strict(
                ddi_df_holdout,
                a.holdout_frac,
                a.seed,
            )

            train_drugs = set(train_df["drug1"].astype(str)).union(
                set(train_df["drug2"].astype(str))
            )
            test_drugs = set(test_df["drug1"].astype(str)).union(
                set(test_df["drug2"].astype(str))
            )
            shared_drugs = train_drugs.intersection(test_drugs)

            logger.info(
                "[drug_holdout_binary] train_drugs=%d test_drugs=%d shared_drugs=%d",
                len(train_drugs),
                len(test_drugs),
                len(shared_drugs),
            )

            if len(train_df) == 0 or len(test_df) == 0:
                raise RuntimeError(
                    "[drug_holdout_binary] empty train/test after split. Adjust holdout_frac or dataset size."
                )

            smiles_train = (
                pd.concat([train_df["smiles_x"], train_df["smiles_y"]])
                .astype(str)
                .unique()
                .tolist()
            )
            tokenizer = SMILESTokenizer(
                smiles_train,
                case_sensitive=a.case_sensitive,
            )

            tr = train_df.copy()
            tr["interaction"] = tr["interaction"].astype(int)

            skf_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=a.seed)
            tr_idx, va_idx = next(skf_inner.split(tr, tr["interaction"]))

            tr_sub = tr.iloc[tr_idx].copy()
            va_sub = tr.iloc[va_idx].copy()

            train_sub_ds = DDIDataset(tr_sub, tokenizer, comp_vecs, d2i, a.max_len)
            val_sub_ds = DDIDataset(va_sub, tokenizer, comp_vecs, d2i, a.max_len)
            test_ds = DDIDataset(test_df, tokenizer, comp_vecs, d2i, a.max_len)

            train_sub_loader = DataLoader(
                train_sub_ds,
                batch_size=a.batch_size,
                shuffle=True,
                collate_fn=custom_collate_fn,
                num_workers=a.num_workers_train,
            )
            val_sub_loader = DataLoader(
                val_sub_ds,
                batch_size=a.batch_size,
                shuffle=False,
                collate_fn=custom_collate_fn,
                num_workers=a.num_workers_val,
            )
            test_loader = DataLoader(
                test_ds,
                batch_size=a.batch_size,
                shuffle=False,
                collate_fn=custom_collate_fn,
                num_workers=a.num_workers_val,
            )

            model = DDIClassifier(
                vocab_size=len(tokenizer.vocab),
                emb_dim=a.embedding_dim,
                d_model=a.d_model,
                graph_dim=graph_dim,
                hid_dim=a.hidden_dim,
                num_classes=2,
                clf_drop=a.classifier_dropout,
                gat_drop=a.gat_dropout,
                lr=a.lr,
                tf_nhead=a.nhead,
                tf_layers=a.num_encoder_layers,
                tf_dim_ff=a.dim_feedforward,
                gat_heads=a.num_heads,
                gat_alpha=a.gat_alpha,
            )

            csv_logger = CSVLogger(a.log_dir, name="drug_holdout_binary")
            early_stopping = EarlyStopping(monitor="val_acc", patience=10, mode="max")
            checkpoint_cb = ModelCheckpoint(
                dirpath=a.model_dir,
                filename="drug_holdout" + "-{epoch:02d}-{val_acc:.4f}",
                monitor="val_acc",
                mode="max",
                save_top_k=1,
            )

            trainer = pl.Trainer(
                max_epochs=a.num_epochs,
                logger=csv_logger,
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                devices=1,
                callbacks=[early_stopping, checkpoint_cb],
                enable_progress_bar=False,
            )
            trainer.fit(model, train_sub_loader, val_sub_loader)

            best_path = checkpoint_cb.best_model_path
            if best_path and os.path.exists(best_path):
                _load_state_dict_safely(model, best_path, device)
                _save_encoder_and_model(model, a.model_dir, "drug_holdout")
            else:
                logger.warning(
                    "[drug_holdout_binary] best checkpoint not found; saving current model weights."
                )
                _save_encoder_and_model(model, a.model_dir, "drug_holdout")

            model.to(device).eval()
            y_true_list: List[int] = []
            y_prob_pos_list: List[float] = []
            y_pred_list: List[int] = []

            with torch.no_grad():
                for t1, t2, v1, v2, y, a1_, a2_ in test_loader:
                    logits = model(
                        t1.to(device),
                        t2.to(device),
                        v1.to(device),
                        v2.to(device),
                        a1_.to(device),
                        a2_.to(device),
                    )
                    probs = torch.softmax(logits, dim=1)[:, 1]
                    preds = torch.argmax(logits, dim=1)

                    y_true_list.extend(y.detach().cpu().tolist())
                    y_prob_pos_list.extend(probs.detach().cpu().tolist())
                    y_pred_list.extend(preds.detach().cpu().tolist())

            y_true = np.asarray(y_true_list, dtype=np.int64)
            y_prob_pos = np.asarray(y_prob_pos_list, dtype=np.float64)

            m = compute_binary_metrics(y_true, y_prob_pos, threshold=0.5)
            rep = classification_report_dict(
                y_true_list,
                y_pred_list,
                digits=4,
                zero_division=1,
            )
            weighted_any = rep.get("weighted avg")
            if not isinstance(weighted_any, dict):
                raise RuntimeError("classification_report missing 'weighted avg'")
            weighted = cast(Dict[str, Any], weighted_any)
            rep_str = classification_report_str(
                y_true_list,
                y_pred_list,
                digits=4,
                zero_division=1,
            )

            out_file = os.path.join(a.result_dir, "drug_holdout_binary_results.txt")
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(f"Shared drugs (train ∩ test): {len(shared_drugs)}\n")
                f.write(f"Holdout frac: {a.holdout_frac}\n")
                f.write(f"Holdout drugs: {len(holdout_drugs)}\n")
                f.write(f"Dropped cross pairs: {dropped}\n\n")
                f.write(f"ACC:   {m['acc']:.4f}\n")
                f.write(f"AUROC: {m['auroc']:.4f}\n")
                f.write(f"AUPRC: {m['auprc']:.4f}\n\n")
                f.write(f"Weighted Precision: {float(weighted['precision']):.4f}\n")
                f.write(f"Weighted Recall:    {float(weighted['recall']):.4f}\n")
                f.write(f"Weighted F1:        {float(weighted['f1-score']):.4f}\n\n")
                f.write(rep_str)

            logger.info(
                "[drug_holdout_binary] acc=%.4f auroc=%.4f auprc=%.4f",
                m["acc"],
                m["auroc"],
                m["auprc"],
            )

    finally:
        shutdown_nvml(nvml_enabled)
