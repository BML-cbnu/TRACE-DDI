# TRACE-DDI

TRACE-DDI is a Transformer–GAT hybrid framework for drug–drug interaction (DDI) prediction.  
It integrates SMILES-based Transformer encoders, graph attention (GAT) networks, and pre-computed compound vectors (e.g., knowledge-graph embeddings).

This repository provides a modular implementation (`trace-ddi.py` + `utils/*`) that reproduces the original experiment structure—generating per-fold logs, checkpoints, and evaluation reports with the same timing and naming conventions.

---

## Environment Setup

**Requirements**
- Python ≥ 3.9 (tested on 3.10–3.12)
- PyTorch ≥ 2.1 (CUDA recommended)
- PyTorch Lightning ≥ 2.0
- scikit-learn, pandas, numpy, matplotlib
- (Optional) `pynvml` for GPU monitoring

---

## Data Format

See `/preprocessing` for details.

---

## Repository Structure

```bash
repo-root/
│
├── trace-ddi.py                  # Main launcher
└── utils/
    ├── __init__.py
    ├── data.py                   # Data loading, tokenization, adjacency, dataset
    ├── model.py                  # Transformer, GAT, and classifier modules
    └── train_eval.py             # Stratified K-Fold training, evaluation, saving


⸻

Best-Parameter Command

python trace-ddi.py \
  --num_epochs 100 \
  --batch_size 32 \
  --lr 2.831e-05 \
  --log_dir path/to/your/result_TRACE/log/ \
  --model_save_dir path/to/your/result_TRACE/savedModel/ \
  --result_dir path/to/your/result_TRACE/ \
  --ddi_data_path path/to/your/data/ddi_01.tsv \
  --smiles_data_path path/to/your/data/smiles_01.tsv \
  --compound_vector_path path/to/your/data/vec20_conv.csv \
  --embedding_dim 256 \
  --d_model 128 \
  --nhead 8 \
  --num_encoder_layers 6 \
  --dim_feedforward 3072 \
  --hidden_dim 256 \
  --classifier_dropout 0.05 \
  --gat_dropout 0.0107 \
  --gat_alpha 0.3738

Important Notes
	•	Use --nhead (not --num_heads) for multi-head attention. Both are accepted, but --nhead is canonical.
	•	Tensor Core GPUs (e.g., RTX 4090) automatically set torch.set_float32_matmul_precision('high') for faster training.
	•	All logs, checkpoints, and reports are saved per fold.

⸻

## Key Arguments

| Argument | Description | Default |
|-----------|-------------|----------|
| `--num_epochs` | Number of training epochs | `100` |
| `--batch_size` | Batch size | `32` |
| `--lr` | Learning rate | `0.00076` |
| `--log_dir` | Directory for logs and plots | `./logs/` |
| `--model_save_dir` | Directory for checkpoints | `./saved_models/` |
| `--result_dir` | Directory for evaluation reports | `./results/` |
| `--ddi_data_path` | Path to DDI TSV | *(required)* |
| `--smiles_data_path` | Path to SMILES TSV | *(required)* |
| `--compound_vector_path` | Path to compound vector CSV | *(required)* |
| `--embedding_dim` | SMILES embedding dimension | `64` |
| `--d_model` | Transformer model dimension | `128` |
| `--nhead` | Number of attention heads | `4` |
| `--num_encoder_layers` | Number of Transformer encoder layers | `3` |
| `--dim_feedforward` | Feed-forward (FFN) dimension | `512` |
| `--hidden_dim` | Hidden dimension of classifier | `256` |
| `--classifier_dropout` | Dropout rate in classifier head | `0.0` |
| `--gat_dropout` | Dropout rate in GAT layer | `0.0145` |
| `--gat_alpha` | Negative slope of GAT LeakyReLU | `0.3086` |
| `--case_sensitive` | Preserve case in SMILES tokens | `off` |
| `--n_splits` | Number of cross-validation folds | `5` |

⸻

Pipeline Overview
	1.	Logging & Reproducibility
	•	Logging initialized at INFO level.
	2.	Data Preparation (utils/data.py)
	•	Load DDI and SMILES tables.
	•	Encode interaction labels.
	•	Tokenize SMILES via regex-based tokenizer.
	•	Construct fixed-size adjacency matrices.
	3.	Model Composition (utils/model.py)
	•	Transformer encoder for SMILES (no positional encoding).
	•	Multi-head GAT with decayed sinusoidal positional encoding.
	•	Combine SMILES + GAT + compound vectors → classification head.
	4.	Training & Evaluation (utils/train_eval.py)
	•	Stratified K-Fold split (default 5 folds).
	•	Label distribution visualization per fold.
	•	EarlyStopping & ModelCheckpoint on val_acc.
	•	Save encoder/full checkpoints and evaluation reports.

⸻

Example Outputs

result_TRACE/
├── log/
├── savedModel/
└── results_fold1.txt

Each results_foldk.txt includes:
	•	Accuracy, Precision, Recall, and F1 (weighted average)
	•	Full classification_report from scikit-learn

⸻

Tips & Troubleshooting
	•	Ensure that drug IDs match across ddi_*.tsv, smiles_*.tsv, and the index of vec*.csv.
	•	For large datasets, increase num_workers in DataLoader to speed up preprocessing.
	•	Any torchmetrics pkg_resources warnings are harmless and safely ignored.
	•	GPU memory usage and temperature can be monitored via pynvml.

⸻