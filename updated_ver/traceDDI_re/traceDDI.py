from __future__ import annotations

import argparse

from utils.preprocess import build_binary_dataset
from utils.trainer import TrainArgs, train_main


def main() -> None:
    p = argparse.ArgumentParser(prog="traceDDI.py")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_b = sub.add_parser("build-binary")
    p_b.add_argument("--pos_pairs_path", type=str, required=True)
    p_b.add_argument("--smiles_path", type=str, required=True)
    p_b.add_argument("--vec_path", type=str, required=True)
    p_b.add_argument("--out_path", type=str, required=True)
    p_b.add_argument("--neg_ratio", type=float, default=1.0)
    p_b.add_argument("--seed", type=int, default=42)
    p_b.add_argument("--pair_dedup", action="store_true")

    p_t = sub.add_parser("train")
    p_t.add_argument("--ddi_path", type=str, required=True)
    p_t.add_argument("--smiles_path", type=str, required=True)
    p_t.add_argument("--vec_path", type=str, required=True)

    p_t.add_argument("--log_dir", type=str, default="./logs")
    p_t.add_argument("--model_dir", type=str, default="./saved_models")
    p_t.add_argument("--result_dir", type=str, default="./results")

    p_t.add_argument(
        "--eval_mode",
        type=str,
        default="multiclass_cv",
        choices=["multiclass_cv", "drug_holdout_binary"],
    )
    p_t.add_argument("--pair_dedup", action="store_true")
    p_t.add_argument("--build_vocab_from_train", action="store_true")
    p_t.add_argument("--holdout_frac", type=float, default=0.2)

    p_t.add_argument("--num_epochs", type=int, default=100)
    p_t.add_argument("--batch_size", type=int, default=32)
    p_t.add_argument("--lr", type=float, default=0.00076)
    p_t.add_argument("--seed", type=int, default=42)
    p_t.add_argument("--max_len", type=int, default=128)
    p_t.add_argument("--n_splits", type=int, default=5)

    p_t.add_argument("--d_model", type=int, default=128)
    p_t.add_argument("--embedding_dim", type=int, default=64)
    p_t.add_argument("--nhead", type=int, default=4)
    p_t.add_argument("--num_encoder_layers", type=int, default=3)
    p_t.add_argument("--dim_feedforward", type=int, default=512)

    p_t.add_argument("--hidden_dim", type=int, default=256)
    p_t.add_argument("--classifier_dropout", type=float, default=0.0)

    p_t.add_argument("--gat_dropout", type=float, default=0.0145)
    p_t.add_argument("--gat_alpha", type=float, default=0.3086)
    p_t.add_argument("--num_heads", type=int, default=8)

    p_t.add_argument("--case_sensitive", action="store_true")
    p_t.add_argument("--num_workers_train", type=int, default=4)
    p_t.add_argument("--num_workers_val", type=int, default=2)

    args = p.parse_args()

    if args.cmd == "build-binary":
        build_binary_dataset(
            pos_pairs_path=args.pos_pairs_path,
            smiles_path=args.smiles_path,
            vec_path=args.vec_path,
            out_path=args.out_path,
            neg_ratio=args.neg_ratio,
            seed=args.seed,
            pair_dedup=args.pair_dedup,
        )
        return

    if args.cmd == "train":
        train_main(
            TrainArgs(
                ddi_path=args.ddi_path,
                smiles_path=args.smiles_path,
                vec_path=args.vec_path,
                log_dir=args.log_dir,
                model_dir=args.model_dir,
                result_dir=args.result_dir,
                eval_mode=args.eval_mode,
                pair_dedup=args.pair_dedup,
                build_vocab_from_train=args.build_vocab_from_train,
                holdout_frac=args.holdout_frac,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                seed=args.seed,
                max_len=args.max_len,
                n_splits=args.n_splits,
                d_model=args.d_model,
                embedding_dim=args.embedding_dim,
                nhead=args.nhead,
                num_encoder_layers=args.num_encoder_layers,
                dim_feedforward=args.dim_feedforward,
                hidden_dim=args.hidden_dim,
                classifier_dropout=args.classifier_dropout,
                gat_dropout=args.gat_dropout,
                gat_alpha=args.gat_alpha,
                num_heads=args.num_heads,
                case_sensitive=args.case_sensitive,
                num_workers_train=args.num_workers_train,
                num_workers_val=args.num_workers_val,
            )
        )
        return


if __name__ == "__main__":
    main()
