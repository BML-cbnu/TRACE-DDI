# Analysis

This directory provides a **complete and modular pipeline** for subgraph-based analysis of drug–drug interactions (DDIs).  
The main script is `CS_analysis.py`, which identifies **shared pathway candidates** between interacting drugs, performs **statistical tests** (MWU + BH-FDR + Cliff’s δ), and saves all resulting statistics and visualizations in a structured output tree.

---

## Contents

    analysis/
    ├── CS_analysis.py           # Main analysis script
    └── README.md                # Documentation

---

## Requirements

- Python ≥ 3.9  
- Required: `pandas`, `numpy`, `networkx`, `matplotlib`  
- Optional: `seaborn`, `scipy`, `pyyaml`, `tqdm`  
- Without SciPy, MWU uses a permutation-like approximation.  
- Without seaborn, KDE plots are skipped while other figures render normally.

**Installation**

    pip install pandas numpy networkx matplotlib seaborn scipy pyyaml tqdm

---

## Input Files

- **DDI table** – `ddi.tsv`  
  TSV file with integer columns: `drug1`, `drug2`, `interaction`

- **Subgraph base directory** – `subG_modify/`  
  Contains:  
  - `edges/compound{drug_id}_edges.tsv`  
  - `nodes/compound{drug_id}_nodes.tsv`

- **Optional mapping resources**
  - `drkg/nodes.tsv`: DRKG node_num → node_name  
  - `hetionet/nodes.tsv`: DRKG node_id → human-readable name  

Use absolute paths to avoid ambiguity.

---

## Output Structure

    <save_root>/
    ├── _run_info/manifest.json
    ├── filtered_wilcoxon/
    │   ├── rank_all.csv
    │   ├── stats/strict/primary_signals.csv
    │   └── viz/strict/A<anchor>/*.png
    ├── simple_wilcoxon/
    │   ├── stats/wilcoxon_results.csv
    │   └── figs/volcano_*.png
    └── extended/
        ├── rank_all_ext.csv
        ├── stats/strict/primary_signals_ext.csv
        └── viz/strict/A<anchor>/*.png

---

## Running the Script

### 1. Direct paths

    python CS_analysis.py \
      --ddi            /REVIEW_ROOT/data/ddi.tsv \
      --subg_base      /REVIEW_ROOT/subG_modify \
      --save_root      /REVIEW_ROOT/visualization/viz \
      --drkg_nodes     /REVIEW_ROOT/drkg/nodes.tsv \
      --hetionet_nodes /REVIEW_ROOT/hetionet/nodes.tsv

---

### 2. Using `--review_root`

**Directory layout**

    /REVIEW_ROOT/
    ├── data/ddi.tsv
    ├── subG_modify/
    ├── drkg/nodes.tsv
    └── hetionet/nodes.tsv

**Command**

    python CS_analysis.py --review_root /REVIEW_ROOT --save_root /REVIEW_ROOT/visualization/viz

---

### 3. Using YAML configuration

You can optionally provide a YAML file (for example `config.review.yaml`) that collects all paths in one place.

Example YAML:

    paths:
      ddi: /REVIEW_ROOT/data/ddi.tsv
      subg_base: /REVIEW_ROOT/subG_modify
      save_root: /REVIEW_ROOT/visualization/viz
      drkg_nodes: /REVIEW_ROOT/drkg/nodes.tsv
      hetionet_nodes: /REVIEW_ROOT/hetionet/nodes.tsv

Example command:

    python CS_analysis.py --config path/to/config.review.yaml

---

### 4. Using environment variables

| Argument        | Environment Variable |
|----------------|----------------------|
| `ddi`          | `CS_DDI`             |
| `subg_base`    | `CS_SUBG_BASE`       |
| `save_root`    | `CS_SAVE_ROOT`       |
| `drkg_nodes`   | `CS_DRKG_NODES`      |
| `hetionet_nodes` | `CS_HETIONET_NODES` |

**Example**

    export CS_DDI=/REVIEW_ROOT/data/ddi.tsv
    export CS_SUBG_BASE=/REVIEW_ROOT/subG_modify
    export CS_SAVE_ROOT=/REVIEW_ROOT/visualization/viz
    export CS_DRKG_NODES=/REVIEW_ROOT/drkg/nodes.tsv
    export CS_HETIONET_NODES=/REVIEW_ROOT/hetionet/nodes.tsv
    python CS_analysis.py

---

## Path Resolution and Reproducibility

Path priority:

1. Command-line arguments (CLI)  
2. Environment variables (ENV)  
3. YAML configuration (`--config`)  
4. `--review_root`  
5. Internal defaults

Each run saves a manifest at `<save_root>/_run_info/manifest.json`, recording all resolved paths and key parameters.

---

## Analysis Workflow

1. **Anchor extraction**  
   Identify `(anchor A, label Y)` pairs from the DDI table and construct anchor-centric interaction sets.

2. **Shared pathway selection**  
   Extract pathway-type nodes shared between A and its interacting partners (`B_int`) with a minimum coverage ratio, using subgraphs under `subG_modify/`.

3. **Centrality contrast**  
   Compute node-level centralities (degree, betweenness, closeness, eigenvector) on merged(A,B) graphs and compare distributions between `B_int` and `B_non` groups.

4. **Ranking and screening**  
   Rank candidate pathways by weighted separation scores, coverage, and eigenvector median; apply:
   - Mann–Whitney U test (MWU),
   - Benjamini–Hochberg FDR (BH-FDR),
   - Cliff’s δ effect size,
   to obtain statistically robust signals.

5. **Visualization**  
   Generate KDE, mean-shift, and merged-graph plots for selected pathways.  
   If `hetionet/nodes.tsv` is supplied, figures use human-readable pathway names.

---

## Main Parameters (defaults)

- `MIN_GROUP_SIZE = 5`  
- `TOPK_PER_ANCHOR = 5`  
- `MIN_SHARE_RATIO = 0.10`  
- `MWU_ALTERNATIVE = "greater"`  
- `FDR_THR = 0.05`  
- `CLIFFS_DELTA_THR = 0.10`  
- Bootstrap iterations: `2000`  
- Random seed: `42`

These defaults are chosen to balance statistical stability and coverage across anchors.

---

## Performance Notes

- In-memory caching for edges, nodes, and graphs prevents redundant disk access across anchors and pathways.  
- `matplotlib.use("Agg")` is enabled for headless rendering on servers or cluster environments.  
- For large `subG_modify/` directories:
  - Place data on a local or fast network filesystem.
  - Consider running on multiple anchors in parallel at the job-scheduling level.

---

## FAQ

**KDE plots not generated?**  
→ Install `seaborn` to enable kernel density plots. Other figures still render without it.

**SciPy missing?**  
→ MWU falls back to a permutation-like approximation. Install `scipy` for the exact implementation.

**Empty or very small outputs?**  
→ Relax `MIN_GROUP_SIZE` or `MIN_SHARE_RATIO`, or check that your DDI table has sufficient interactions per anchor.

**Unreadable labels (IDs only)?**  
→ Provide `--hetionet_nodes` so that DRKG node IDs can be mapped to human-readable pathway names in all figures.