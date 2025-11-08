"""
TRACE-DDI main launcher.
- Imports modular components from utils/*.
- Produces per-fold logs, checkpoints, and metrics.
"""

import os
import argparse
import warnings
import pytorch_lightning as pl

from utils.data import load_and_prepare_data
from utils.train_eval import run_stratified_cv

def build_argparser():
    p = argparse.ArgumentParser()

    # Experiment parameters
    p.add_argument('--num_epochs', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr', type=float, default=0.00076)
    p.add_argument('--log_dir', type=str, default='./logs/')
    p.add_argument('--model_save_dir', type=str, default='./saved_models/')
    p.add_argument('--result_dir', type=str, default='./results/')
    p.add_argument('--ddi_data_path', type=str, required=True)
    p.add_argument('--smiles_data_path', type=str, required=True)
    p.add_argument('--compound_vector_path', type=str, required=True)

    # Transformer parameters
    p.add_argument('--d_model', type=int, default=128)
    p.add_argument('--embedding_dim', type=int, default=64)
    p.add_argument('--nhead', type=int, default=4)
    p.add_argument('--num_encoder_layers', type=int, default=3)
    p.add_argument('--dim_feedforward', type=int, default=512)

    # Classifier parameters
    p.add_argument('--hidden_dim', type=int, default=256)
    p.add_argument('--classifier_dropout', type=float, default=0.0)

    # GAT parameters
    p.add_argument('--gat_dropout', type=float, default=0.0145)
    p.add_argument('--gat_alpha', type=float, default=0.3086)
    p.add_argument('--num_heads', type=int, default=12)

    # Tokenizer case sensitivity
    p.add_argument('--case_sensitive', action='store_true',
                   help='Keep SMILES tokenization case-sensitive')

    # CV control
    p.add_argument('--n_splits', type=int, default=5)
    return p


def main():

    import logging, sys
    logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
    stream=sys.stdout,
    force=True
    )
    
    # Silence torchmetrics pkg_resources warning
    warnings.filterwarnings("ignore", category=UserWarning, module="torchmetrics.utilities.imports")

    args = build_argparser().parse_args()

    # Create dirs (no informational logs here; real logs occur inside utils)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_save_dir, exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)

    # Reproducibility
    pl.seed_everything(42, workers=True)

    # Load data & tokenizer (logs happen inside)
    (ddi_df, tokenizer, code_to_label,
     compound_vectors, drug_to_vector_index) = load_and_prepare_data(
        ddi_path=args.ddi_data_path,
        smiles_path=args.smiles_data_path,
        compound_vec_path=args.compound_vector_path,
        case_sensitive=args.case_sensitive
    )

    # Run K-fold training/evaluation (logs happen inside)
    run_stratified_cv(
        ddi_df=ddi_df,
        tokenizer=tokenizer,
        code_to_label=code_to_label,
        compound_vectors=compound_vectors,
        drug_to_vector_index=drug_to_vector_index,
        args=args
    )


if __name__ == '__main__':
    main()