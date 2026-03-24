import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

# ---- Paths
DDI_PATH_DEFAULT = "./ddi/data/tsv/ddi_pairs.tsv"
SUBG_BASE_DEFAULT = "./ddi/data/subgraphs/preprocessed_kg_subgraphs"
SAVE_ROOT_DEFAULT = "./ddi/data/results"
DRKG_NODES_DEFAULT = "./ddi/data/tsv/drkg_nodes.tsv"
HETIONET_NODES_DEFAULT = "./ddi/data/tsv/hetionet_filtered_nodes.tsv"

# ---- Toggles
SAVE_TOGGLE = True

# ---- Core hyperparameters
MIN_GROUP_SIZE = 5
TOPK_PER_ANCHOR = 5
GLOBAL_TOPK = 10
NON_RATIO = 2.0
MIN_SHARE_RATIO = 0.10

REP_W = {
    "eigenvector": 0.3,
    "betweenness": 0.4,
    "degree": 0.2,
    "closeness": 0.1,
}

EIGEN_USE_POWER = True
EIG_MAX_ITER = 500
EIG_TOL = 1e-6

N_ANCHORS_FOR_REPORT = 5
FDR_THR = 0.05
CLIFFS_DELTA_THR = 0.10
KEY_METRICS = ("degree", "betweenness", "closeness", "eigenvector")
TOP_N_MAIN = 10

MWU_ALTERNATIVE = "greater"
BOOT_N = 2000
RNG_SEED = 42

METRICS = ("degree", "betweenness", "closeness", "eigenvector")

# ---- Extended metrics
EXTRA_METRICS = ("clustering_inv", "eccentricity_inv", "radiality")
METRICS_EXT = ("degree", "betweenness", "closeness", "eigenvector") + EXTRA_METRICS

REP_W_EXT = {
    "betweenness": 0.40,
    "eigenvector": 0.30,
    "degree": 0.20,
    "closeness": 0.10,
    "radiality": 0.08,
    "clustering_inv": 0.07,
    "eccentricity_inv": 0.05,
}

# ---- Cases
CASE_FILTERED = "filtered_wilcoxon"
CASE_SIMPLE = "simple_wilcoxon"
CASE_EXTENDED = "extended"

# ---- Plot defaults
plt.rcParams["figure.dpi"] = 120
plt.rcParams["axes.grid"] = False
plt.rcParams["font.size"] = 10
sns.set_style("whitegrid")
