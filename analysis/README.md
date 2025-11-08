⸻

Analysis

This directory provides a complete and modular pipeline for subgraph-based analysis of drug–drug interactions (DDIs).
The main script is CS_analysis.py, which identifies shared pathway candidates between interacting drugs, performs statistical tests (MWU + BH-FDR + Cliff’s δ), and saves all resulting statistics and visualizations in a single output tree.

⸻

Contents

analysis/
├── CS_analysis.py                 # Main analysis script
├── README.md                      # Documentation
└── examples/
    ├── config.review.yaml         # Example configuration
    └── run_examples.md            # Example execution guide


⸻

Requirements
	•	Python ≥ 3.9
	•	Required: pandas, numpy, networkx, matplotlib
	•	Optional: seaborn, scipy, pyyaml, tqdm
	•	Without SciPy, MWU uses a permutation-like approximation.
	•	Without seaborn, KDE plots are skipped while all other figures render normally.

Installation:

pip install pandas numpy networkx matplotlib seaborn scipy pyyaml tqdm


⸻

Input Files
	•	DDI table – ddi.tsv
TSV with integer columns: drug1, drug2, interaction
	•	Subgraph base directory – subG_modify/
	•	edges/compound{drug_id}_edges.tsv
	•	nodes/compound{drug_id}_nodes.tsv
	•	Optional mapping resources
	•	drkg/nodes.tsv: DRKG node_num → node_name
	•	hetionet/nodes.tsv: DRKG node_id → human-readable name
(e.g., filtered-hetionet-v1.0-nodes.tsv renamed to hetionet/nodes.tsv)

Use absolute paths to avoid path resolution conflicts.

⸻

Output Structure

All results are written under --save_root:

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


⸻

Running the Script

1. Direct paths

python CS_analysis.py \
  --ddi            /REVIEW_ROOT/data/ddi.tsv \
  --subg_base      /REVIEW_ROOT/subG_modify \
  --save_root      /REVIEW_ROOT/visualization/viz \
  --drkg_nodes     /REVIEW_ROOT/drkg/nodes.tsv \
  --hetionet_nodes /REVIEW_ROOT/hetionet/nodes.tsv

2. Using --review_root

If the directory follows the layout:

/REVIEW_ROOT/
├── data/ddi.tsv
├── subG_modify/
├── drkg/nodes.tsv
└── hetionet/nodes.tsv

Run:

python CS_analysis.py --review_root /REVIEW_ROOT --save_root /REVIEW_ROOT/visualization/viz

3. Using YAML configuration

analysis/examples/config.review.yaml

paths:
  ddi: /REVIEW_ROOT/data/ddi.tsv
  subg_base: /REVIEW_ROOT/subG_modify
  save_root: /REVIEW_ROOT/visualization/viz
  drkg_nodes: /REVIEW_ROOT/drkg/nodes.tsv
  hetionet_nodes: /REVIEW_ROOT/hetionet/nodes.tsv

Run:

python CS_analysis.py --config analysis/examples/config.review.yaml

4. Environment variables

Argument	Environment variable
ddi	CS_DDI
subg_base	CS_SUBG_BASE
save_root	CS_SAVE_ROOT
drkg_nodes	CS_DRKG_NODES
hetionet_nodes	CS_HETIONET_NODES

Example:

export CS_DDI=/REVIEW_ROOT/data/ddi.tsv
export CS_SUBG_BASE=/REVIEW_ROOT/subG_modify
export CS_SAVE_ROOT=/REVIEW_ROOT/visualization/viz
python CS_analysis.py


⸻

Path Resolution and Reproducibility

Path priority: CLI > ENV > YAML > --review_root > defaults.
Every run records a complete manifest in <save_root>/_run_info/manifest.json.

⸻

Analysis Workflow
	1.	Anchor extraction – Identify (anchor A, label Y) pairs from the DDI table.
	2.	Shared pathway selection – Find pathway-type nodes common to A and its interacting partners (B_int) using a minimum coverage ratio.
	3.	Centrality contrast – Compute node-level centralities (degree, betweenness, closeness, eigenvector) on merged(A,B) graphs; compare B_int vs B_non.
	4.	Ranking and screening – Score by weighted separation + coverage + centrality; filter using MWU p-value, BH-FDR, and Cliff’s δ.
	5.	Visualization – Generate KDE, mean-shift, and merged-graph plots.
	•	hetionet/nodes.tsv provides readable labels for all graph figures.

FAQ

KDE plots not generated?
Install seaborn (pip install seaborn).

SciPy missing?
MWU uses an internal approximation if SciPy is unavailable.

Empty output folders?
Relax MIN_GROUP_SIZE or MIN_SHARE_RATIO.

Unreadable node labels?
Provide a valid --hetionet_nodes mapping file.

⸻