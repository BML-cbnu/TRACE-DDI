# Updated TRACE-DDI

This repository contains the updated implementation of TRACE-DDI for drug-drug interaction (DDI) prediction.

## Overview

TRACE-DDI is a multimodal framework that integrates:

- SMILES sequence representations
- SMILES-derived molecular graph representations
- Knowledge graph-based drug vectors

The code supports the following core functionalities:

- multiclass DDI prediction
- strict drug-holdout binary evaluation
- binary dataset construction from positive interaction pairs
- modular training pipeline with separated utilities for preprocessing, tokenization, graph construction, dataset handling, model definition, and training

## Project Structure

```text
traceDDI/
├── traceDDI.py
├── utils/
│   ├── __init__.py
│   ├── common.py
│   ├── metrics.py
│   ├── preprocess.py
│   ├── tokenizer.py
│   ├── graph.py
│   ├── dataset.py
│   ├── model.py
│   └── trainer.py
├── data/
├── logs/
├── saved_models/
├── results/
└── requirements.txt
```

## Main Features

- sequence encoding of SMILES strings using a regex-based tokenizer and Transformer encoder
- graph-based encoding of molecular structure through SMILES-derived adjacency construction and graph attention layers
- integration of external drug vectors for complementary biological context
- cross-validation-based multiclass evaluation
- strict inductive evaluation through drug-holdout binary splitting
- automatic binary dataset generation from known positive pairs
- checkpoint, log, and result management through a modular training interface

## Notes

- Parameter settings and experiment-specific options are documented separately.
- Paths should be adapted to the local environment before execution.
