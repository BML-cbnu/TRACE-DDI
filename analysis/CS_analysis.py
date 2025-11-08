"""
CS_analysis.py — distributable DDI subgraph analysis
Key features:
- Robust path resolution (CLI > ENV > YAML --config > --review_root > defaults)
- Pure-Python, no project-specific imports (PyData + networkx + optional SciPy/pyyaml)
- Save-only orchestration: writes CSV/PNG artifacts under save_root
"""

# =========================
# Part 0. Imports & Globals
# =========================
import os, re, sys, json, argparse, warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import networkx as nx

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

# seaborn is optional; if missing, fall back to matplotlib-only
try:
    import seaborn as sns
    sns.set_style("whitegrid")
except Exception:
    sns = None

from typing import Tuple, Optional, Dict, Iterable, List, Set

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **k): return x

plt.rcParams["figure.dpi"] = 120
plt.rcParams["axes.grid"] = False
plt.rcParams["font.size"]  = 10

# ---- Default paths (safe, generic fallbacks; can be overridden)
DDI_PATH_DEFAULT   = "./data/ddi.tsv"
SUBG_BASE_DEFAULT  = "./subG_modify"
SAVE_ROOT_DEFAULT  = "./visualization/viz"
DRKG_NODES_DEFAULT = "./drkg/nodes.tsv"
HETNET_NODES_DEFAULT = "./hetionet/nodes.tsv"

# ---- Environment variable keys (for portability in clusters/CI)
ENV_KEYS = {
    "ddi": "CS_DDI",
    "subg_base": "CS_SUBG_BASE",
    "save_root": "CS_SAVE_ROOT",
    "drkg_nodes": "CS_DRKG_NODES",
    "hetionet_nodes": "CS_HETIONET_NODES",
}

# ---- Core hyperparameters
SAVE_TOGGLE       = True
MIN_GROUP_SIZE    = 5
TOPK_PER_ANCHOR   = 5
GLOBAL_TOPK       = 10
NON_RATIO         = 2.0
MIN_SHARE_RATIO   = 0.10

REP_W = {"eigenvector": 0.3, "betweenness": 0.4, "degree": 0.2, "closeness": 0.1}
EIGEN_USE_POWER = True
EIG_MAX_ITER = 500
EIG_TOL      = 1e-6

N_ANCHORS_FOR_REPORT = 5
FDR_THR          = 0.05
CLIFFS_DELTA_THR = 0.10
KEY_METRICS      = ("degree", "betweenness", "closeness", "eigenvector")
TOP_N_MAIN       = 10

MWU_ALTERNATIVE  = "greater"
BOOT_N           = 2000
RNG_SEED         = 42

METRICS = ("degree","betweenness","closeness","eigenvector")

# Extended metrics
EXTRA_METRICS   = ("clustering_inv","eccentricity_inv","radiality")
METRICS_EXT     = ("degree","betweenness","closeness","eigenvector") + EXTRA_METRICS
REP_W_EXT = {
    "betweenness":0.40, "eigenvector":0.30, "degree":0.20, "closeness":0.10,
    "radiality":0.08, "clustering_inv":0.07, "eccentricity_inv":0.05
}

# Folder cases
CASE_FILTERED = "filtered_wilcoxon"
CASE_SIMPLE   = "simple_wilcoxon"
CASE_EXTENDED = "extended"

# =========================
# Part 0+. Path Resolution
# =========================
def _read_yaml(path: Optional[str]) -> dict:
    """Load YAML if available; return {} otherwise."""
    if not path: return {}
    try:
        import yaml
    except Exception:
        return {}
    try:
        with open(path, "r") as f:
            cfg = yaml.safe_load(f) or {}
        return cfg if isinstance(cfg, dict) else {}
    except Exception:
        return {}

def _pick_value(key: str, args_ns, cfg: dict, review_root_map: dict, default_val: str) -> str:
    """Priority: CLI > ENV > YAML > review_root inference > default."""
    # CLI
    v_cli = getattr(args_ns, key, None)
    if v_cli not in (None, ""):
        return str(v_cli)

    # ENV
    env_key = ENV_KEYS.get(key)
    if env_key and os.getenv(env_key):
        return os.getenv(env_key)

    # YAML (flat or under "paths")
    if key in cfg and isinstance(cfg[key], (str, int)):
        return str(cfg[key])
    if isinstance(cfg.get("paths"), dict) and cfg["paths"].get(key):
        return str(cfg["paths"][key])

    # review_root inference
    if review_root_map.get(key):
        return str(review_root_map[key])

    # default
    return str(default_val)

def resolve_paths(args) -> Dict[str, str]:
    """
    Build a path map:
      - If --review_root is given, infer standard subfolders:
            data/ddi.tsv, subG_modify/, visualization/viz/, drkg/nodes.tsv, hetionet/nodes.tsv
      - Apply priority merging and return resolved absolute paths (dirs normalized).
    """
    rr = os.path.abspath(args.review_root) if getattr(args, "review_root", None) else None
    inferred = {}
    if rr:
        inferred = {
            "ddi": os.path.join(rr, "data", "ddi.tsv"),
            "subg_base": os.path.join(rr, "subG_modify"),
            "save_root": os.path.join(rr, "visualization", "viz"),
            "drkg_nodes": os.path.join(rr, "drkg", "nodes.tsv"),
            "hetionet_nodes": os.path.join(rr, "hetionet", "nodes.tsv"),
        }

    cfg = _read_yaml(getattr(args, "config", None))

    resolved = {
        "ddi":           _pick_value("ddi", args, cfg, inferred, DDI_PATH_DEFAULT),
        "subg_base":     _pick_value("subg_base", args, cfg, inferred, SUBG_BASE_DEFAULT),
        "save_root":     _pick_value("save_root", args, cfg, inferred, SAVE_ROOT_DEFAULT),
        "drkg_nodes":    _pick_value("drkg_nodes", args, cfg, inferred, DRKG_NODES_DEFAULT),
        "hetionet_nodes":_pick_value("hetionet_nodes", args, cfg, inferred, HETNET_NODES_DEFAULT),
    }

    # Normalize; ensure dirs are creatable
    resolved["ddi"]        = os.path.abspath(resolved["ddi"])
    resolved["subg_base"]  = os.path.abspath(resolved["subg_base"])
    resolved["save_root"]  = os.path.abspath(resolved["save_root"])
    resolved["drkg_nodes"] = os.path.abspath(resolved["drkg_nodes"]) if resolved["drkg_nodes"] else ""
    resolved["hetionet_nodes"] = os.path.abspath(resolved["hetionet_nodes"]) if resolved["hetionet_nodes"] else ""

    # Make save_root
    os.makedirs(resolved["save_root"], exist_ok=True)

    # Save manifest for reproducibility
    try:
        info_dir = os.path.join(resolved["save_root"], "_run_info")
        os.makedirs(info_dir, exist_ok=True)
        manifest = {
            "paths": resolved,
            "args": {k: (v if isinstance(v,(str,int,float,bool)) else str(v)) for k,v in vars(args).items()},
            "env": {k: os.getenv(k) for k in ENV_KEYS.values() if os.getenv(k)}
        }
        with open(os.path.join(info_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)
    except Exception:
        pass

    return resolved

# =========================
# Part 1. Utilities & I/O
# =========================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def safe_fname(txt: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", txt)

def short_label(name: str, maxlen: int=36) -> str:
    s = str(name)
    if len(s) <= maxlen: return s
    if s.startswith("Pathway::"):
        head = "Pathway::"; tail = s.split("::", 1)[1]
        if len(tail) > maxlen - len(head):
            tail = tail[:maxlen-len(head)-3] + "..."
        return head + tail
    return s[:maxlen-3] + "..."

def read_ddi(path: str) -> pd.DataFrame:
    """Load DDI TSV: three integer cols (drug1, drug2, interaction)."""
    df = pd.read_csv(path, sep="\t", header=None,
                     names=["drug1","drug2","interaction"],
                     dtype={"drug1":"Int64", "drug2":"Int64", "interaction":"Int64"})
    for c in ("drug1","drug2","interaction"):
        df[c] = df[c].astype(int, copy=False)
    return df

def _assert_ddi_types(ddi: pd.DataFrame):
    """Validate DDI dtypes (fail fast in review settings)."""
    assert pd.api.types.is_integer_dtype(ddi["drug1"])
    assert pd.api.types.is_integer_dtype(ddi["drug2"])
    assert pd.api.types.is_integer_dtype(ddi["interaction"])

# ---- Output layout
class OutPaths:
    """Consistent tree under save_root: <case>/{stats,viz,figs}/..."""
    def __init__(self, save_root: str):
        self.root = save_root; ensure_dir(self.root)
    def case_root(self, case: str) -> str:
        p = os.path.join(self.root, case); ensure_dir(p); return p
    def stats_dir(self, case: str, stage: Optional[str]=None) -> str:
        base = os.path.join(self.case_root(case), "stats")
        if stage: base = os.path.join(base, stage)
        ensure_dir(base); return base
    def viz_dir(self, case: str, stage: Optional[str]=None) -> str:
        base = os.path.join(self.case_root(case), "viz")
        if stage: base = os.path.join(base, stage)
        ensure_dir(base); return base
    def figs_dir(self, case: str, sub: Optional[str]=None) -> str:
        base = os.path.join(self.case_root(case), "figs")
        if sub: base = os.path.join(base, sub)
        ensure_dir(base); return base

# =========================
# Part 2. Subgraph Cache I/O
# =========================
_EDGES_NODES_CACHE: Dict[Tuple[int,str], Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]] = {}
_GRAPH_CACHE: Dict[Tuple[int,int], nx.Graph] = {}
_CENTRALITY_CACHE: Dict[Tuple[int,int], Dict[str,Dict[str,float]]] = {}
_PATHWAY_SET_CACHE: Dict[Tuple[int,str], Set[str]] = {}

def load_graph_files(drug_id: int, base_dir: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Read per-drug edges/nodes TSVs from subG_modify structure."""
    efile = os.path.join(base_dir, f"edges/compound{int(drug_id)}_edges.tsv")
    nfile = os.path.join(base_dir, f"nodes/compound{int(drug_id)}_nodes.tsv")
    if not (os.path.exists(efile) and os.path.exists(nfile)):
        return None, None
    edges = pd.read_csv(efile, sep="\t", header=None,
                        names=["node1","relation_id","node2","ori_node1_num","ori_node2_num","node1_name","node2_name"])
    try:
        nodes = pd.read_csv(nfile, sep="\t", header=None,
                            names=["node_id","node_name","node_type","visited_count","spread_value","ori_node_id"])
    except Exception:
        nodes = pd.read_csv(nfile, sep="\t", header=None,
                            names=["node_id","node_name","node_type","visited_count","spread_value","ori_nude_id"])
        nodes = nodes.rename(columns={"ori_nude_id": "ori_node_id"})
    return edges, nodes

def _std_nodes(nodes: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Normalize node_name field for robust membership tests."""
    if nodes is None: return None
    out = nodes.copy()
    out["node_name"] = out["node_name"].astype(str).str.strip()
    return out

def load_graph_files_cached(drug_id: int, base_dir: str):
    """Cache wrapper for edges/nodes."""
    key = (int(drug_id), base_dir)
    if key in _EDGES_NODES_CACHE:
        return _EDGES_NODES_CACHE[key]
    e, n = load_graph_files(int(drug_id), base_dir)
    n = _std_nodes(n)
    _EDGES_NODES_CACHE[key] = (e, n)
    return e, n

def get_pathway_set_cached(drug_id: int, base_dir: str) -> Set[str]:
    """Set of pathway-like node names for a drug subgraph."""
    key = (int(drug_id), base_dir)
    if key in _PATHWAY_SET_CACHE:
        return _PATHWAY_SET_CACHE[key]
    _, n = load_graph_files_cached(int(drug_id), base_dir)
    if n is None:
        s: Set[str] = set()
    else:
        s = set(n.loc[(n["node_type"] == 9) | (n["node_name"].astype(str).str.startswith("Pathway::")),
                      "node_name"].astype(str))
    _PATHWAY_SET_CACHE[key] = s
    return s

def merged_graph_edges(Ae: Optional[pd.DataFrame], Be: Optional[pd.DataFrame]) -> nx.Graph:
    """Merge two edge tables into an undirected graph with color tags."""
    G = nx.Graph()
    def _add(edges, color):
        if edges is None: return
        for _, r in edges.iterrows():
            u, v = str(r["node1_name"]), str(r["node2_name"])
            G.add_edge(u, v, relation=r["relation_id"], color=color)
    _add(Ae, "#1E88E5"); _add(Be, "#43A047")
    return G

def merged_graph_cached(Ae, Be, A_id: int, B_id: int) -> nx.Graph:
    """Cache merged graph per (A,B)."""
    key = (int(A_id), int(B_id))
    if key in _GRAPH_CACHE:
        return _GRAPH_CACHE[key]
    G = merged_graph_edges(Ae, Be)
    _GRAPH_CACHE[key] = G
    return G

def centralities_for_all_nodes_cached(G: nx.Graph, A_id: int, B_id: int) -> Dict[str,Dict[str,float]]:
    """Cache centralities per merged graph."""
    key = (int(A_id), int(B_id))
    if key in _CENTRALITY_CACHE:
        return _CENTRALITY_CACHE[key]
    deg = nx.degree_centrality(G)
    btw = nx.betweenness_centrality(G)
    clo = nx.closeness_centrality(G)
    if EIGEN_USE_POWER:
        try:
            eig = nx.eigenvector_centrality(G, max_iter=EIG_MAX_ITER, tol=EIG_TOL)
        except nx.PowerIterationFailedConvergence:
            eig = {n: 0.0 for n in G.nodes()}
    else:
        try:
            eig = nx.eigenvector_centrality_numpy(G)
        except Exception:
            eig = {n: 0.0 for n in G.nodes()}
    out = {"degree":deg, "betweenness":btw, "closeness":clo, "eigenvector":eig}
    _CENTRALITY_CACHE[key] = out
    return out

# =========================
# Part 3. Name Resolution
# =========================
class NameResolver:
    """Map numeric IDs to readable names (DRKG/Hetionet if available)."""
    def __init__(self, drkg_nodes_path: Optional[str] = None, hetionet_nodes_path: Optional[str] = None):
        self.drkg = None; self.hetn = None
        if drkg_nodes_path and os.path.exists(drkg_nodes_path):
            self.drkg = pd.read_csv(drkg_nodes_path, sep="\t", names=['node_num', 'node_name', 'node_type'])
        if hetionet_nodes_path and os.path.exists(hetionet_nodes_path):
            self.hetn = pd.read_csv(hetionet_nodes_path, sep="\t")
        self._drkg_num2name = {}
        if self.drkg is not None and 'node_num' in self.drkg and 'node_name' in self.drkg:
            self._drkg_num2name = dict(self.drkg[['node_num','node_name']].values)
        self._hetn_id2name = {}
        if self.hetn is not None and 'id' in self.hetn.columns and 'name' in self.hetn.columns:
            self._hetn_id2name = dict(self.hetn[['id','name']].values)

    def drug_label(self, x: int) -> str:
        """Prefer Hetionet name via DRKG node_name → Hetionet name; fallback to raw."""
        try:
            if self.drkg is not None:
                node_name = self._drkg_num2name.get(int(x), None)
                if node_name:
                    if self.hetn is not None:
                        return self._hetn_id2name.get(str(node_name), str(node_name))
                    return str(node_name)
            return str(x)
        except Exception:
            return str(x)

    def pathway_label(self, node_info) -> str:
        """Return readable pathway label if known."""
        try:
            if isinstance(node_info, int) and self.drkg is not None:
                node_name = self._drkg_num2name.get(int(node_info), None)
            else:
                node_name = str(node_info)
            if self.hetn is not None and node_name is not None:
                return self._hetn_id2name.get(str(node_name), str(node_name))
            return str(node_name) if node_name is not None else "Unknown"
        except Exception:
            return str(node_info)

# =========================
# Part 4. DDI helpers
# =========================
def partners_of(ddi: pd.DataFrame, drug_id: int) -> pd.DataFrame:
    """Rows where drug_id participates."""
    drug_id = int(drug_id)
    return ddi[(ddi["drug1"] == drug_id) | (ddi["drug2"] == drug_id)].copy()

def calculate_portion(ddi_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-interaction top co-occurring drugs (≥50% portion)."""
    rows = []
    for interaction, g in ddi_df.groupby("interaction", sort=False):
        counts = g["drug1"].value_counts().add(g["drug2"].value_counts(), fill_value=0)
        portions = (counts / (len(g)*2)) * 100
        sel = portions[portions >= 50]
        rows.extend({"interaction": int(interaction), "drug": int(d), "portion": float(p)}
                    for d, p in sel.items())
    return pd.DataFrame(rows)

def B_sets_for_label(ddi: pd.DataFrame, anchor_a: int, label_y: int) -> Tuple[List[int], List[int]]:
    """Return (interacting-Bs, non-interacting-Bs) for anchor A and label Y."""
    anchor_a = int(anchor_a); label_y = int(label_y)
    ab = partners_of(ddi, anchor_a)
    mask = ((ab["drug1"]==anchor_a) & (ab["interaction"]==label_y)) | ((ab["drug2"]==anchor_a) & (ab["interaction"]==label_y))
    Bs_y = set(ab.loc[mask, ["drug1","drug2"]].to_numpy().ravel()); Bs_y.discard(anchor_a)
    all_drugs = set(ddi["drug1"]).union(set(ddi["drug2"]))
    return sorted(map(int, Bs_y)), sorted(map(int, all_drugs - Bs_y - {anchor_a}))

def drug_has_pathway(drug_id: int, base_dir: str, pathway: str) -> bool:
    """True if B's subgraph contains the pathway node."""
    return pathway in get_pathway_set_cached(int(drug_id), base_dir)

def pathway_candidates_shared(anchor_a: int, label_y: int, base_dir: str, ddi: pd.DataFrame,
                              min_share_ratio: float = MIN_SHARE_RATIO, max_expand_B: int = 200):
    """Pathways shared between A and a sufficient fraction of B_int drugs."""
    _, An = load_graph_files_cached(int(anchor_a), base_dir)
    if An is None:
        return set(), {}
    A_paths = get_pathway_set_cached(int(anchor_a), base_dir)
    B_int, _ = B_sets_for_label(ddi, int(anchor_a), int(label_y))
    if not B_int:
        return set(), {}
    pool = B_int[:max_expand_B]
    cnt: Dict[str,int] = {}
    for b in pool:
        shared = A_paths & get_pathway_set_cached(int(b), base_dir)
        for x in shared:
            cnt[x] = cnt.get(x, 0) + 1
    cand, cov = set(), {}
    nB = max(1, len(pool))
    for x, c in cnt.items():
        ratio = c / nB
        if ratio >= min_share_ratio:
            cand.add(x); cov[x] = (c, ratio)
    return cand, cov

# =========================
# Part 5. Measurements & Stats
# =========================
def measure_X_over_B_fast(anchor_a: int, pathway: str, B_list: Iterable[int], base_dir: str,
                          require_shared: bool = True) -> pd.DataFrame:
    """Centralities of pathway node in merged(A,B) graphs across B_list."""
    Ae, _ = load_graph_files_cached(int(anchor_a), base_dir)
    if Ae is None:
        return pd.DataFrame(columns=["drug_b", *METRICS])
    out = []
    for b in B_list:
        b = int(b)
        if require_shared and not drug_has_pathway(b, base_dir, pathway):
            continue
        Be, _ = load_graph_files_cached(b, base_dir)
        if Be is None:
            continue
        G = merged_graph_cached(Ae, Be, int(anchor_a), b)
        c = centralities_for_all_nodes_cached(G, int(anchor_a), b)
        if pathway not in c["degree"]:
            continue
        out.append({
            "drug_b": b,
            "degree":      c["degree"].get(pathway, 0.0),
            "betweenness": c["betweenness"].get(pathway, 0.0),
            "closeness":   c["closeness"].get(pathway, 0.0),
            "eigenvector": c["eigenvector"].get(pathway, 0.0),
        })
    return pd.DataFrame(out)

def separation_score(dist_df: pd.DataFrame, weights: Dict[str,float]=REP_W) -> Dict[str,float]:
    """Weighted median-gap between interacting vs non-interacting."""
    out = {}
    gi = dist_df["group"]=="interacting"; gn = dist_df["group"]=="non-interacting"
    for m in METRICS:
        med_i = float(np.nanmedian(dist_df.loc[gi, m])) if gi.any() else 0.0
        med_n = float(np.nanmedian(dist_df.loc[gn, m])) if gn.any() else 0.0
        out[f"diff_med_{m}"] = med_i - med_n
    out["sep_score"] = sum(weights[k]*out[f"diff_med_{k}"] for k in weights)
    return out

def sample_non_set(non_list: List[int], target_len: int) -> List[int]:
    """Deterministic downsample for fair contrasts."""
    if len(non_list) <= target_len:
        return list(map(int, non_list))
    rng = np.random.default_rng(RNG_SEED)
    idx = rng.choice(len(non_list), size=target_len, replace=False)
    return [int(non_list[i]) for i in idx]

# Wilcoxon helpers (SciPy optional)
try:
    from scipy.stats import mannwhitneyu
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

def mwu_pvalue(a: np.ndarray, b: np.ndarray, alternative: str = MWU_ALTERNATIVE) -> float:
    """MWU with fallback permutation-like test."""
    if len(a) == 0 or len(b) == 0: return 1.0
    if _HAVE_SCIPY:
        try:
            return float(mannwhitneyu(a, b, alternative=alternative).pvalue)
        except Exception:
            pass
    rng = np.random.default_rng(0)
    allv = np.concatenate([a, b]); obs = np.median(a) - np.median(b)
    cnt = 0; iters = 2000
    for _ in range(iters):
        rng.shuffle(allv); aa = allv[:len(a)]; bb = allv[len(a):]
        if (np.median(aa) - np.median(bb)) >= obs:
            cnt += 1
    return (cnt + 1) / (iters + 1)

def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Effect size (probability of superiority)."""
    a = np.asarray(a); b = np.asarray(b)
    na, nb = len(a), len(b)
    if na == 0 or nb == 0: return 0.0
    gt = sum(x > y for x in a for y in b)
    lt = sum(x < y for x in a for y in b)
    return (gt - lt) / float(na * nb)

def benjamini_hochberg(pvals: List[float]) -> List[float]:
    """BH-FDR correction (vectorized)."""
    m = len(pvals); idx = np.argsort(pvals); ranked = np.array(pvals)[idx]
    q = np.empty(m, dtype=float); prev = 1.0
    for i in range(m-1, -1, -1):
        rank = i + 1; val = (m / rank) * ranked[i]
        prev = min(prev, val); q[i] = prev
    out = np.empty(m, dtype=float); out[idx] = q
    return out.tolist()

def summarize_metric_stats(dist_df: pd.DataFrame, alternative: str = MWU_ALTERNATIVE) -> Dict[str, Dict[str, float]]:
    """Compute per-metric d_mean / d_median / p / δ."""
    res = {}
    gi = dist_df["group"]=="interacting"; gn = dist_df["group"]=="non-interacting"
    for m in METRICS:
        a = dist_df.loc[gi, m].dropna().to_numpy()
        b = dist_df.loc[gn, m].dropna().to_numpy()
        dm = float(np.nanmean(a) - np.nanmean(b)) if len(a) and len(b) else 0.0
        dmed = float(np.nanmedian(a) - np.nanmedian(b)) if len(a) and len(b) else 0.0
        p = mwu_pvalue(a, b, alternative=alternative)
        d = cliffs_delta(a, b)
        res[m] = {"d_mean": dm, "d_median": dmed, "p_mwu": p, "cliffs_delta": d}
    return res

# =========================
# Part 6. Ranking & Screening
# =========================
def evaluate_anchor_label_fast(ddi: pd.DataFrame, anchor_a: int, label_y: int, base_dir: str,
                               min_group=MIN_GROUP_SIZE, top_k_pathways=TOPK_PER_ANCHOR,
                               non_ratio=NON_RATIO, min_share_ratio: float = MIN_SHARE_RATIO) -> pd.DataFrame:
    """Rank candidate pathways for (A,Y) by separation score + coverage."""
    B_int, B_non = B_sets_for_label(ddi, int(anchor_a), int(label_y))
    if (len(B_int) < min_group) or (len(B_non) < min_group):
        return pd.DataFrame()
    candX, coverage = pathway_candidates_shared(int(anchor_a), int(label_y), base_dir, ddi, min_share_ratio=min_share_ratio)
    if not candX:
        return pd.DataFrame()
    non_target_len = int(min(len(B_non), max(min_group, non_ratio * len(B_int))))
    B_non_s = sample_non_set(B_non, non_target_len)

    rows = []
    for X in candX:
        df_i = measure_X_over_B_fast(int(anchor_a), X, B_int, base_dir, require_shared=True)
        df_n = measure_X_over_B_fast(int(anchor_a), X, B_non_s, base_dir, require_shared=True)
        if df_i.empty or df_n.empty:
            continue
        df_i["group"] = "interacting"; df_n["group"] = "non-interacting"
        dist = pd.concat([df_i, df_n], ignore_index=True)
        sep = separation_score(dist, weights=REP_W)
        med_eig_int = float(np.nanmedian(df_i["eigenvector"])) if "eigenvector" in df_i else 0.0
        cov_count, cov_ratio = coverage.get(X, (0, 0.0))
        final = sep["sep_score"] * (med_eig_int if med_eig_int > 0 else 1.0) * (1.0 + cov_ratio)
        rows.append({
            "anchor": int(anchor_a), "interaction": int(label_y), "pathway": X,
            "n_int": int(len(df_i)), "n_non": int(len(df_n)),
            "coverage_count": int(cov_count), "coverage_ratio": float(cov_ratio),
            **sep, "median_eig_int": float(med_eig_int), "final_score": float(final)
        })

    res = pd.DataFrame(rows)
    if res.empty:
        return res
    return res.sort_values("final_score", ascending=False).head(top_k_pathways).reset_index(drop=True)

def identify_signal_pairs(rank_all: pd.DataFrame,
                          ddi: pd.DataFrame,
                          base_dir: str,
                          delta_thr: float = CLIFFS_DELTA_THR,
                          fdr_thr: float = FDR_THR,
                          min_each_group: int = MIN_GROUP_SIZE,
                          min_coverage: float = MIN_SHARE_RATIO,
                          require_two_metrics: Tuple[str, ...] = KEY_METRICS,
                          top_n_primary: int = TOP_N_MAIN,
                          enable_fallback_when_empty: bool = True,
                          use_or_on_keys: bool = False,
                          min_n_sig_metrics: int = 2,
                          mwu_alternative: str = MWU_ALTERNATIVE) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Strict screening (BH-FDR + δ) with relaxed and fallback modes."""
    tmp_pvals, per_row_metrics = [], []
    for A, y, X in rank_all[["anchor","interaction","pathway"]].itertuples(index=False):
        A = int(A); y = int(y); X = str(X)
        B_int, B_non = B_sets_for_label(ddi, A, y)
        B_int_shared = [int(b) for b in B_int if drug_has_pathway(int(b), base_dir, X)]
        B_non_shared = [int(b) for b in B_non if drug_has_pathway(int(b), base_dir, X)]

        cov_ok = float(rank_all.loc[(rank_all["anchor"]==A)&(rank_all["interaction"]==y)&(rank_all["pathway"]==X),"coverage_ratio"].iloc[0]) >= min_coverage
        size_ok = (len(B_int_shared) >= min_each_group) and (len(B_non_shared) >= min_each_group)
        if not (cov_ok and size_ok):
            continue

        df_i = measure_X_over_B_fast(A, X, B_int_shared, base_dir, require_shared=True)
        df_n = measure_X_over_B_fast(A, X, B_non_shared, base_dir, require_shared=True)
        if df_i.empty or df_n.empty:
            continue
        df_i["group"] = "interacting"; df_n["group"] = "non-interacting"
        dist_df = pd.concat([df_i, df_n], ignore_index=True)
        stats = summarize_metric_stats(dist_df, alternative=mwu_alternative)
        per_row_metrics.append(((A, y, X), stats, len(df_i), len(df_n), float(rank_all.loc[(rank_all["anchor"]==A)&(rank_all["interaction"]==y)&(rank_all["pathway"]==X),"coverage_ratio"].iloc[0])))
        for m in METRICS:
            tmp_pvals.append(stats[m]["p_mwu"])

    base_cols   = ["anchor","interaction","pathway","n_int","n_non","coverage_ratio","is_primary","selection_stage","notes"]
    metric_cols = sum(([f"{m}_q", f"{m}_d_mean", f"{m}_delta"] for m in METRICS), [])
    extra_cols  = ["n_sig_metrics", "sum_d_mean", "min_q"]

    def _empty_frames():
        sig_df = pd.DataFrame(columns=base_cols + metric_cols + extra_cols)
        primary_df = sig_df.copy()
        return primary_df, sig_df

    if not tmp_pvals:
        return _empty_frames()

    qvals = benjamini_hochberg(tmp_pvals)
    qi = 0; sig_rows = []
    for (A, y, X), stats, n_i, n_n, cov in per_row_metrics:
        mstats = {}; min_q_val = 1.0; sum_dmean = 0.0
        for m in METRICS:
            p = stats[m]["p_mwu"]; q = qvals[qi]; qi += 1
            dm = stats[m]["d_mean"]; d = stats[m]["cliffs_delta"]
            mstats[m] = {"p": p, "q": q, "d_mean": dm, "cliffs_delta": d}
            min_q_val = min(min_q_val, q); sum_dmean += max(0.0, dm)

        sig_count = sum(
            (mstats[k]["q"] < fdr_thr) and
            (mstats[k]["cliffs_delta"] >= delta_thr) and
            (mstats[k]["d_mean"] > 0.0)
            for k in require_two_metrics
        )
        pass_keys = (sig_count >= 1) if use_or_on_keys else (sig_count >= min_n_sig_metrics)

        if pass_keys:
            sig_rows.append({
                "anchor": int(A), "interaction": int(y), "pathway": X,
                "n_int": int(n_i), "n_non": int(n_n), "coverage_ratio": float(cov),
                **{f"{m}_q": float(mstats[m]["q"]) for m in METRICS},
                **{f"{m}_d_mean": float(mstats[m]["d_mean"]) for m in METRICS},
                **{f"{m}_delta": float(mstats[m]["cliffs_delta"]) for m in METRICS},
                "n_sig_metrics": int(sig_count),
                "sum_d_mean": float(sum_dmean),
                "min_q": float(min_q_val),
                "is_primary": False, "selection_stage": "strict", "notes": f"[strict] k-of-{len(require_two_metrics)}: {sig_count} met"
            })

    if len(sig_rows) > 0:
        sig_df = (pd.DataFrame(sig_rows)
                  .sort_values(["n_sig_metrics","sum_d_mean","min_q"], ascending=[False, False, True]))
        primary_df = sig_df.head(TOP_N_MAIN).copy(); primary_df.loc[:, "is_primary"] = True
        return primary_df, sig_df

    # Relaxed
    LOOSE_P_THR   = 0.10
    LOOSE_DELTA   = 0.05
    MIN_ABS_DMEAN = {"degree": 0.001, "betweenness": 1e-5, "closeness": 0.0005, "eigenvector": 1e-4}
    loose_rows = []
    for (A, y, X), stats, n_i, n_n, cov in per_row_metrics:
        passed_any = False; mstats = {}; sum_dmean = 0.0; n_passed = 0
        for m in METRICS:
            p = stats[m]["p_mwu"]; dm = stats[m]["d_mean"]; d = stats[m]["cliffs_delta"]
            cond = (dm > MIN_ABS_DMEAN[m]) and (d >= LOOSE_DELTA) and (p < LOOSE_P_THR)
            if cond: n_passed += 1
            passed_any = passed_any or cond
            mstats[m] = {"q": p, "d_mean": dm, "cliffs_delta": d}
            sum_dmean += max(0.0, dm)
        if passed_any and (n_i >= MIN_GROUP_SIZE) and (n_n >= MIN_GROUP_SIZE) and (cov >= MIN_SHARE_RATIO):
            loose_rows.append({
                "anchor": int(A), "interaction": int(y), "pathway": X,
                "n_int": int(n_i), "n_non": int(n_n), "coverage_ratio": float(cov),
                **{f"{m}_q": float(mstats[m]["q"]) for m in METRICS},
                **{f"{m}_d_mean": float(mstats[m]["d_mean"]) for m in METRICS},
                **{f"{m}_delta": float(mstats[m]["cliffs_delta"]) for m in METRICS},
                "n_sig_metrics": int(n_passed),
                "sum_d_mean": float(sum_dmean),
                "min_q": float(min([mstats[m]["q"] for m in METRICS])),
                "is_primary": False, "selection_stage": "relaxed",
                "notes": "[relaxed] p<0.10 & d_mean>min & δ≥0.05"
            })

    if len(loose_rows) > 0:
        sig_df = (pd.DataFrame(loose_rows)
                  .sort_values(["n_sig_metrics","sum_d_mean","min_q"], ascending=[False, False, True]))
        primary_df = sig_df.head(TOP_N_MAIN).copy(); primary_df.loc[:, "is_primary"] = True
        return primary_df, sig_df

    # Fallback
    if not enable_fallback_when_empty:
        empty = pd.DataFrame(columns=base_cols + metric_cols + extra_cols)
        return empty, empty
    fallback = (rank_all.sort_values("final_score", ascending=False)
                        [["anchor","interaction","pathway"]]
                        .drop_duplicates().head(TOP_N_MAIN).copy())
    for c in ["n_int","n_non","coverage_ratio","is_primary","selection_stage","notes","n_sig_metrics","sum_d_mean","min_q"]:
        fallback[c] = np.nan
    for m in METRICS:
        fallback[f"{m}_q"] = np.nan; fallback[f"{m}_d_mean"] = np.nan; fallback[f"{m}_delta"] = np.nan
    fallback["is_primary"] = False
    fallback["selection_stage"] = "fallback"
    fallback["notes"] = "[fallback] ranked by final_score; not statistically selected"
    return fallback.copy(), fallback.copy()

# =========================
# Part 7. Simple Wilcoxon
# =========================
def run_simple_wilcoxon(rank_all: pd.DataFrame,
                        ddi: pd.DataFrame,
                        base_dir: str,
                        min_each_group: int = MIN_GROUP_SIZE,
                        mwu_alternative: str = MWU_ALTERNATIVE) -> pd.DataFrame:
    """Plain MWU for all (A,Y,X) combos."""
    if rank_all.empty:
        return pd.DataFrame()
    rows = []; all_pvals = []; stash = []
    for A, y, X in rank_all[["anchor","interaction","pathway"]].drop_duplicates().itertuples(index=False):
        A, y, X = int(A), int(y), str(X)
        B_int, B_non = B_sets_for_label(ddi, A, y)
        B_int_shared = [int(b) for b in B_int if drug_has_pathway(int(b), base_dir, X)]
        B_non_shared = [int(b) for b in B_non if drug_has_pathway(int(b), base_dir, X)]
        if (len(B_int_shared) < min_each_group) or (len(B_non_shared) < min_each_group):
            continue
        df_i = measure_X_over_B_fast(A, X, B_int_shared, base_dir, require_shared=True)
        df_n = measure_X_over_B_fast(A, X, B_non_shared, base_dir, require_shared=True)
        if df_i.empty or df_n.empty:
            continue
        df_i["group"] = "interacting"; df_n["group"] = "non-interacting"
        dist_df = pd.concat([df_i, df_n], ignore_index=True)
        stats = summarize_metric_stats(dist_df, alternative=mwu_alternative)
        cov_ratio_series = rank_all.loc[(rank_all["anchor"]==A)&(rank_all["interaction"]==y)&(rank_all["pathway"]==X),"coverage_ratio"]
        cov_ratio = float(cov_ratio_series.iloc[0]) if len(cov_ratio_series) else np.nan
        stash.append(((A,y,X), stats, int((dist_df["group"]=="interacting").sum()),
                      int((dist_df["group"]=="non-interacting").sum()), cov_ratio))
        for m in METRICS:
            all_pvals.append(stats[m]["p_mwu"])
    if not stash:
        return pd.DataFrame()
    qvals = benjamini_hochberg(all_pvals); qi = 0
    for (A,y,X), stats, n_i, n_n, cov_ratio in stash:
        row = {"anchor":A, "interaction":y, "pathway":X, "n_int":n_i, "n_non":n_n, "coverage_ratio":cov_ratio}
        min_q = 1.0; n_p_lt_005 = 0
        for m in METRICS:
            p = float(stats[m]["p_mwu"]); q = float(qvals[qi]); qi += 1
            d_mean = float(stats[m]["d_mean"]); d_median = float(stats[m]["d_median"])
            row[f"{m}_p"] = p; row[f"{m}_q"] = q; row[f"{m}_d_mean"] = d_mean; row[f"{m}_d_median"] = d_median
            min_q = min(min_q, q); n_p_lt_005 += int(p < 0.05)
        row["min_q"] = float(min_q); row["n_p_lt_0_05"] = int(n_p_lt_005)
        rows.append(row)
    out = pd.DataFrame(rows).sort_values(["min_q","n_p_lt_0_05"], ascending=[True, False]).reset_index(drop=True)
    return out

# =========================
# Part 8. Extended Metrics
# =========================
def _radiality_per_component(G: nx.Graph, nodes: Iterable[str], dist_map: Dict[str, Dict[str,int]]) -> Dict[str, float]:
    """Radiality within a connected component."""
    sub = G.subgraph(nodes)
    if sub.number_of_nodes() <= 1:
        return {n: 1.0 for n in sub.nodes()}
    max_d = 1
    for v in sub.nodes():
        for u,d in dist_map[v].items():
            if u in sub and u != v and d > max_d:
                max_d = d
    D = max(1, max_d)
    out = {}
    for v in sub.nodes():
        s = 0.0; cnt = 0
        for u in sub.nodes():
            if u == v:
                continue
            d = dist_map[v].get(u, None)
            if d is None:
                continue
            s += (D + 1 - d); cnt += 1
        out[v] = (s / max(1, cnt)) / D
    return out

def compute_extra_metrics(G: nx.Graph) -> Dict[str, Dict[str, float]]:
    """Clustering^-1, Eccentricity^-1, Radiality."""
    clustering = nx.clustering(G); eps = 1e-9
    clustering_inv = {n: 1.0 / max(eps, v) for n, v in clustering.items()}
    ecc = {}
    for comp in nx.connected_components(G):
        sub = G.subgraph(comp)
        try:
            sub_ecc = nx.eccentricity(sub)
        except Exception:
            spl = dict(nx.all_pairs_shortest_path_length(sub))
            sub_ecc = {v: (max(d.values()) if len(d) else 0) for v, d in spl.items()}
        ecc.update(sub_ecc)
    ecc_inv = {n: (1.0/float(v) if v > 0 else 0.0) for n, v in ecc.items()}
    all_spl = dict(nx.all_pairs_shortest_path_length(G))
    rad = {}
    for comp in nx.connected_components(G):
        rad.update(_radiality_per_component(G, comp, all_spl))
    return {"clustering_inv":clustering_inv, "eccentricity_inv":ecc_inv, "radiality":rad}

def measure_X_over_B_fast_ext(anchor_a: int, pathway: str, B_list: Iterable[int], base_dir: str,
                              require_shared: bool = True) -> pd.DataFrame:
    """Extended metrics across merged(A,B) graphs."""
    Ae, _ = load_graph_files_cached(int(anchor_a), base_dir)
    if Ae is None:
        return pd.DataFrame(columns=["drug_b", *METRICS_EXT])
    out = []
    for b in B_list:
        b = int(b)
        if require_shared and not drug_has_pathway(b, base_dir, pathway):
            continue
        Be, _ = load_graph_files_cached(b, base_dir)
        if Be is None:
            continue
        G = merged_graph_cached(Ae, Be, int(anchor_a), b)
        base = centralities_for_all_nodes_cached(G, int(anchor_a), b)
        extra = compute_extra_metrics(G)
        if pathway not in base["degree"]:
            continue
        out.append({
            "drug_b": b,
            "degree":      base["degree"].get(pathway, 0.0),
            "betweenness": base["betweenness"].get(pathway, 0.0),
            "closeness":   base["closeness"].get(pathway, 0.0),
            "eigenvector": base["eigenvector"].get(pathway, 0.0),
            "clustering_inv":   extra["clustering_inv"].get(pathway, 0.0),
            "eccentricity_inv": extra["eccentricity_inv"].get(pathway, 0.0),
            "radiality":        extra["radiality"].get(pathway, 0.0)
        })
    return pd.DataFrame(out)

def separation_score_ext(dist_df: pd.DataFrame, weights: Dict[str,float]=REP_W_EXT,
                         metrics: Tuple[str,...]=METRICS_EXT) -> Dict[str,float]:
    """Weighted median-gap with extended metrics."""
    out = {}
    gi = dist_df["group"]=="interacting"; gn = dist_df["group"]=="non-interacting"
    for m in metrics:
        med_i = float(np.nanmedian(dist_df.loc[gi, m])) if gi.any() else 0.0
        med_n = float(np.nanmedian(dist_df.loc[gn, m])) if gn.any() else 0.0
        out[f"diff_med_{m}"] = med_i - med_n
    out["sep_score"] = sum(weights.get(k,0.0)*out.get(f"diff_med_{k}",0.0) for k in metrics if k in weights)
    return out

def evaluate_anchor_label_fast_ext(ddi: pd.DataFrame, anchor_a: int, label_y: int, base_dir: str,
                                   min_group=MIN_GROUP_SIZE, top_k_pathways=TOPK_PER_ANCHOR,
                                   non_ratio=NON_RATIO, min_share_ratio: float = MIN_SHARE_RATIO) -> pd.DataFrame:
    """Extended ranking for (A,Y)."""
    B_int, B_non = B_sets_for_label(ddi, int(anchor_a), int(label_y))
    if (len(B_int) < min_group) or (len(B_non) < min_group):
        return pd.DataFrame()
    candX, coverage = pathway_candidates_shared(int(anchor_a), int(label_y), base_dir, ddi, min_share_ratio=min_share_ratio)
    if not candX:
        return pd.DataFrame()
    non_target_len = int(min(len(B_non), max(min_group, non_ratio * len(B_int))))
    B_non_s = sample_non_set(B_non, non_target_len)

    rows = []
    for X in candX:
        df_i = measure_X_over_B_fast_ext(int(anchor_a), X, B_int, base_dir, require_shared=True)
        df_n = measure_X_over_B_fast_ext(int(anchor_a), X, B_non_s, base_dir, require_shared=True)
        if df_i.empty or df_n.empty:
            continue
        df_i["group"] = "interacting"; df_n["group"] = "non-interacting"
        dist = pd.concat([df_i, df_n], ignore_index=True)
        sep = separation_score_ext(dist, weights=REP_W_EXT, metrics=METRICS_EXT)
        med_eig_int = float(np.nanmedian(df_i["eigenvector"])) if "eigenvector" in df_i else 0.0
        cov_count, cov_ratio = coverage.get(X, (0, 0.0))
        final = sep["sep_score"] * (med_eig_int if med_eig_int > 0 else 1.0) * (1.0 + cov_ratio)
        rows.append({
            "anchor": int(anchor_a), "interaction": int(label_y), "pathway": X,
            "n_int": int(len(df_i)), "n_non": int(len(df_n)),
            "coverage_count": int(cov_count), "coverage_ratio": float(cov_ratio),
            **sep, "median_eig_int": float(med_eig_int), "final_score": float(final)
        })
    res = pd.DataFrame(rows)
    if res.empty:
        return res
    return res.sort_values("final_score", ascending=False).head(top_k_pathways).reset_index(drop=True)

def summarize_metric_stats_ext(dist_df: pd.DataFrame, metrics: Tuple[str,...]=METRICS_EXT,
                               alternative: str = MWU_ALTERNATIVE) -> Dict[str, Dict[str, float]]:
    """MWU/δ for extended metrics."""
    res = {}
    gi = dist_df["group"]=="interacting"; gn = dist_df["group"]=="non-interacting"
    for m in metrics:
        a = dist_df.loc[gi, m].dropna().to_numpy()
        b = dist_df.loc[gn, m].dropna().to_numpy()
        dm = float(np.nanmean(a) - np.nanmean(b)) if len(a) and len(b) else 0.0
        dmed = float(np.nanmedian(a) - np.nanmedian(b)) if len(a) and len(b) else 0.0
        p = mwu_pvalue(a, b, alternative=alternative)
        d = cliffs_delta(a, b)
        res[m] = {"d_mean": dm, "d_median": dmed, "p_mwu": p, "cliffs_delta": d}
    return res

def identify_signal_pairs_ext(rank_all: pd.DataFrame,
                              ddi: pd.DataFrame,
                              base_dir: str,
                              metrics: Tuple[str,...]=METRICS_EXT,
                              delta_thr: float = CLIFFS_DELTA_THR,
                              fdr_thr: float = FDR_THR,
                              min_each_group: int = MIN_GROUP_SIZE,
                              min_coverage: float = MIN_SHARE_RATIO,
                              require_two_metrics: Tuple[str, ...] = KEY_METRICS,
                              top_n_primary: int = TOP_N_MAIN,
                              enable_fallback_when_empty: bool = True,
                              use_or_on_keys: bool = False,
                              min_n_sig_metrics: int = 2,
                              mwu_alternative: str = MWU_ALTERNATIVE) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Extended screening (BH-FDR + δ) with relaxed/fallback."""
    tmp_pvals, per_row_metrics = [], []
    for A, y, X in rank_all[["anchor","interaction","pathway"]].itertuples(index=False):
        A = int(A); y = int(y); X = str(X)
        B_int, B_non = B_sets_for_label(ddi, A, y)
        B_int_shared = [int(b) for b in B_int if drug_has_pathway(int(b), base_dir, X)]
        B_non_shared = [int(b) for b in B_non if drug_has_pathway(int(b), base_dir, X)]
        cov_series = rank_all.loc[(rank_all["anchor"]==A)&(rank_all["interaction"]==y)&(rank_all["pathway"]==X),"coverage_ratio"]
        cov_ok = (float(cov_series.iloc[0]) if len(cov_series) else 0.0) >= min_coverage
        size_ok = (len(B_int_shared) >= min_each_group) and (len(B_non_shared) >= min_each_group)
        if not (cov_ok and size_ok):
            continue
        df_i = measure_X_over_B_fast_ext(A, X, B_int_shared, base_dir, require_shared=True)
        df_n = measure_X_over_B_fast_ext(A, X, B_non_shared, base_dir, require_shared=True)
        if df_i.empty or df_n.empty:
            continue
        df_i["group"] = "interacting"; df_n["group"] = "non-interacting"
        dist_df = pd.concat([df_i, df_n], ignore_index=True)
        stats = summarize_metric_stats_ext(dist_df, metrics=metrics, alternative=mwu_alternative)
        per_row_metrics.append(((A, y, X), stats, len(df_i), len(df_n), float(cov_series.iloc[0])))
        for m in metrics:
            tmp_pvals.append(stats[m]["p_mwu"])

    base_cols   = ["anchor","interaction","pathway","n_int","n_non","coverage_ratio","is_primary","selection_stage","notes"]
    metric_cols = sum(([f"{m}_q", f"{m}_d_mean", f"{m}_delta"] for m in metrics), [])
    extra_cols  = ["n_sig_metrics", "sum_d_mean", "min_q"]

    def _empty_frames():
        sig_df = pd.DataFrame(columns=base_cols + metric_cols + extra_cols); primary_df = sig_df.copy(); return primary_df, sig_df

    if not tmp_pvals:
        return _empty_frames()

    qvals = benjamini_hochberg(tmp_pvals)
    qi = 0; sig_rows = []
    for (A, y, X), stats, n_i, n_n, cov in per_row_metrics:
        mstats = {}; min_q_val = 1.0; sum_dmean = 0.0
        for m in metrics:
            p = stats[m]["p_mwu"]; q = qvals[qi]; qi += 1
            dm = stats[m]["d_mean"]; d = stats[m]["cliffs_delta"]
            mstats[m] = {"p": p, "q": q, "d_mean": dm, "cliffs_delta": d}
            min_q_val = min(min_q_val, q); sum_dmean += max(0.0, dm)

        sig_count = sum(
            (mstats[k]["q"] < fdr_thr) and
            (mstats[k]["cliffs_delta"] >= delta_thr) and
            (mstats[k]["d_mean"] > 0.0)
            for k in require_two_metrics if k in mstats
        )
        pass_keys = (sig_count >= 1) if use_or_on_keys else (sig_count >= min_n_sig_metrics)

        if pass_keys:
            sig_rows.append({
                "anchor": int(A), "interaction": int(y), "pathway": X,
                "n_int": int(n_i), "n_non": int(n_n), "coverage_ratio": float(cov),
                **{f"{m}_q": float(mstats[m]["q"]) for m in metrics},
                **{f"{m}_d_mean": float(mstats[m]["d_mean"]) for m in metrics},
                **{f"{m}_delta": float(mstats[m]["cliffs_delta"]) for m in metrics},
                "n_sig_metrics": int(sig_count),
                "sum_d_mean": float(sum_dmean),
                "min_q": float(min_q_val),
                "is_primary": False, "selection_stage": "strict", "notes": f"[strict] k-of-{len(require_two_metrics)}: {sig_count} met"
            })

    if len(sig_rows) > 0:
        sig_df = (pd.DataFrame(sig_rows)
                  .sort_values(["n_sig_metrics","sum_d_mean","min_q"], ascending=[False, False, True]))
        primary_df = sig_df.head(TOP_N_MAIN).copy(); primary_df.loc[:, "is_primary"] = True
        return primary_df, sig_df

    # Relaxed
    LOOSE_P_THR   = 0.10
    LOOSE_DELTA   = 0.05
    MIN_ABS_DMEAN = {
        "degree": 0.001, "betweenness": 1e-5, "closeness": 0.0005, "eigenvector": 1e-4,
        "clustering_inv": 1e-4, "eccentricity_inv": 1e-4, "radiality": 1e-4
    }
    loose_rows = []
    for (A, y, X), stats, n_i, n_n, cov in per_row_metrics:
        passed_any = False; mstats = {}; sum_dmean = 0.0; n_passed = 0
        for m in metrics:
            p = stats[m]["p_mwu"]; dm = stats[m]["d_mean"]; d = stats[m]["cliffs_delta"]
            cond = (dm > MIN_ABS_DMEAN.get(m, 0.0)) and (d >= LOOSE_DELTA) and (p < LOOSE_P_THR)
            if cond: n_passed += 1
            passed_any = passed_any or cond
            mstats[m] = {"q": p, "d_mean": dm, "cliffs_delta": d}
            sum_dmean += max(0.0, dm)
        if passed_any and (n_i >= MIN_GROUP_SIZE) and (n_n >= MIN_GROUP_SIZE) and (cov >= MIN_SHARE_RATIO):
            loose_rows.append({
                "anchor": int(A), "interaction": int(y), "pathway": X,
                "n_int": int(n_i), "n_non": int(n_n), "coverage_ratio": float(cov),
                **{f"{m}_q": float(mstats[m]["q"]) for m in metrics},
                **{f"{m}_d_mean": float(mstats[m]["d_mean"]) for m in metrics},
                **{f"{m}_delta": float(mstats[m]["cliffs_delta"]) for m in metrics},
                "n_sig_metrics": int(n_passed),
                "sum_d_mean": float(sum_dmean),
                "min_q": float(min([mstats[m]["q"] for m in metrics])),
                "is_primary": False, "selection_stage": "relaxed",
                "notes": "[relaxed] p<0.10 & d_mean>min & δ≥0.05"
            })

    if len(loose_rows) > 0:
        sig_df = (pd.DataFrame(loose_rows)
                  .sort_values(["n_sig_metrics","sum_d_mean","min_q"], ascending=[False, False, True]))
        primary_df = sig_df.head(TOP_N_MAIN).copy(); primary_df.loc[:, "is_primary"] = True
        return primary_df, sig_df

    # Fallback
    if not enable_fallback_when_empty:
        empty = pd.DataFrame(columns=base_cols + metric_cols + extra_cols)
        return empty, empty
    fallback = (rank_all.sort_values("final_score", ascending=False)
                        [["anchor","interaction","pathway"]]
                        .drop_duplicates().head(TOP_N_MAIN).copy())
    for c in ["n_int","n_non","coverage_ratio","is_primary","selection_stage","notes","n_sig_metrics","sum_d_mean","min_q"]:
        fallback[c] = np.nan
    for m in metrics:
        fallback[f"{m}_q"] = np.nan; fallback[f"{m}_d_mean"] = np.nan; fallback[f"{m}_delta"] = np.nan
    fallback["is_primary"] = False
    fallback["selection_stage"] = "fallback"
    fallback["notes"] = "[fallback] ranked by final_score; not statistically selected"
    return fallback.copy(), fallback.copy()

# =========================
# Part 9. Visualization
# =========================
def _node_name_to_nodeid_pref_node_id(nodes: Optional[pd.DataFrame]) -> Dict[str, int]:
    """Map node_name → numeric id (prefers node_id)."""
    if nodes is None or nodes.empty:
        return {}
    col = "node_id" if "node_id" in nodes.columns else ("ori_node_id" if "ori_node_id" in nodes.columns else None)
    if col is None:
        return {}
    return {str(r["node_name"]): int(pd.to_numeric(r[col], errors="coerce")) for _, r in nodes.iterrows()}

def build_node_color_map(G: nx.Graph,
                         An: Optional[pd.DataFrame], Bn: Optional[pd.DataFrame],
                         Ae: Optional[pd.DataFrame], Be: Optional[pd.DataFrame],
                         drug_a: int, drug_b: int,
                         highlight: Optional[Iterable[str]] = None) -> Dict[str, str]:
    """Color code: A (blue), B (green), A∩B (purple), pathway (amber), drug nodes (red)."""
    highlight = set(highlight or [])
    mapA = _node_name_to_nodeid_pref_node_id(An)
    mapB = _node_name_to_nodeid_pref_node_id(Bn)
    colors: Dict[str, str] = {}
    for n in G.nodes():
        ns = str(n)
        if ns in highlight:
            colors[n] = "#FFD54F"; continue
        if (ns in mapA and mapA[ns] == int(drug_a)) or (ns in mapB and mapB[ns] == int(drug_b)):
            colors[n] = "#EF5350"; continue
        inA, inB = ns in mapA, ns in mapB
        colors[n] = "#5E35B1" if (inA and inB) else ("#1E88E5" if inA else ("#43A047" if inB else "#9E9E9E"))
    return colors

def draw_merged_graph(drug_a: int, drug_b: int, base_dir: str, save_path: str,
                      resolver: NameResolver, highlight_pathway: Optional[str]=None,
                      seed: int=42, figsize=(12,8), size_by_eig: bool=False, centrality_box: bool=True,
                      interaction_y: Optional[int]=None):
    """Merged(A,B) network with optional highlighted pathway."""
    Ae, An = load_graph_files_cached(int(drug_a), base_dir)
    Be, Bn = load_graph_files_cached(int(drug_b), base_dir)
    if Ae is None or Be is None:
        return
    G = merged_graph_cached(Ae, Be, int(drug_a), int(drug_b))
    colors = build_node_color_map(G, An, Bn, Ae, Be, drug_a, drug_b,
                                  highlight=[highlight_pathway] if highlight_pathway else None)
    pos = nx.spring_layout(G, seed=seed)

    if size_by_eig:
        try:
            eig = nx.eigenvector_centrality(G, max_iter=EIG_MAX_ITER, tol=EIG_TOL)
            vals = np.array([eig.get(n,0.0) for n in G.nodes()])
            vals = 300 + 600*(vals/vals.max()) if vals.max() > 0 else np.full_like(vals, 500.0)
        except nx.PowerIterationFailedConvergence:
            vals = np.full(len(G.nodes()), 500.0)
    else:
        vals = np.full(len(G.nodes()), 520.0)

    plt.figure(figsize=figsize)
    nx.draw(
        G, pos, with_labels=True,
        labels={n: short_label(n) for n in G.nodes()},
        node_color=[colors[n] for n in G.nodes()],
        edge_color=[d["color"] for _,_,d in G.edges(data=True)],
        node_size=vals, font_size=9, width=1.1
    )

    if centrality_box and highlight_pathway and highlight_pathway in G:
        cents = centralities_for_all_nodes_cached(G, int(drug_a), int(drug_b))
        info = "\n".join([
            f"degree: {cents['degree'].get(highlight_pathway,0):.4f}",
            f"betweenness: {cents['betweenness'].get(highlight_pathway,0):.4f}",
            f"closeness: {cents['closeness'].get(highlight_pathway,0):.4f}",
            f"eigenvector: {cents['eigenvector'].get(highlight_pathway,0):.4f}",
        ])
        plt.text(0.98, 0.98, info, transform=plt.gca().transAxes,
                 ha="right", va="top", fontsize=9,
                 bbox=dict(facecolor="white", alpha=0.7, edgecolor="#BDBDBD"))

    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor="#EF5350", label="Drug (A/B, exact)"),
        Patch(facecolor="#1E88E5", label="A subgraph"),
        Patch(facecolor="#43A047", label="B subgraph"),
        Patch(facecolor="#5E35B1", label="A∩B"),
        Patch(facecolor="#FFD54F", label="Highlighted pathway"),
    ]
    plt.legend(handles=legend_elems, loc="lower left", framealpha=0.95)

    ttl = f"Merged — A={resolver.drug_label(drug_a)} + B={resolver.drug_label(drug_b)}"
    if highlight_pathway: ttl += f"\n| highlight: {resolver.pathway_label(highlight_pathway)}"
    if interaction_y is not None: ttl += f" | Y={int(interaction_y)}"
    plt.title(ttl, pad=20)
    plt.tight_layout()
    if SAVE_TOGGLE and save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def kde_with_top_markers(dist_df: pd.DataFrame, pathway: str, anchor_a: int,
                         interaction_y: int, resolver: NameResolver,
                         save_path: Optional[str]=None, stage: Optional[str] = None,
                         metrics: Tuple[str, ...] = METRICS):
    """KDE for chosen metrics (defaults to 4)."""
    if sns is None:  # fallback
        return
    chosen = tuple(m for m in metrics if m in METRICS)
    if len(chosen) == 0:
        return
    n = len(chosen); ncols = 2; nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 5*nrows if n>2 else 6))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n:]: ax.set_visible(False)

    n_int = int((dist_df["group"]=="interacting").sum())
    n_non = int((dist_df["group"]=="non-interacting").sum())
    labels = {"interacting": f"interacting (n={n_int})", "non-interacting": f"non-interacting (n={n_non})"}

    for i, m in enumerate(chosen):
        ax = axes[i]
        for g, color in [("interacting","red"), ("non-interacting","blue")]:
            sub = dist_df[dist_df["group"]==g]
            if len(sub):
                sns.kdeplot(data=sub, x=m, fill=True, alpha=0.3, linewidth=2, color=color, ax=ax, label=labels[g])
        if len(dist_df):
            thr = dist_df[m].quantile(0.95)
            ax.axvline(thr, color="black", linestyle=":", linewidth=1.2)
            ax.text(thr, ax.get_ylim()[1]*0.9, "95th %", rotation=90, va="top", ha="right", fontsize=9)
        ax.set_title(m.capitalize()); ax.set_xlabel("Centrality"); ax.set_ylabel("Density"); ax.legend()

    title = f"A={resolver.drug_label(anchor_a)} | Y={int(interaction_y)} | Pathway={resolver.pathway_label(pathway)}"
    if stage: title = f"[{stage}] " + title
    fig.suptitle(title, fontsize=12)
    plt.tight_layout(rect=[0,0,1,0.93])
    if SAVE_TOGGLE and save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def _bootstrap_mean_diff(a: np.ndarray, b: np.ndarray, n_boot: int = BOOT_N, seed: int = RNG_SEED) -> Tuple[float,float]:
    """Bootstrap CI for mean difference."""
    if len(a) == 0 or len(b) == 0: return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    ia = rng.integers(0, len(a), size=(n_boot, len(a)))
    ib = rng.integers(0, len(b), size=(n_boot, len(b)))
    diffs = np.nanmean(a[ia], axis=1) - np.nanmean(b[ib], axis=1)
    lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
    return float(lo), float(hi)

def mean_shift_panel(dist_df: pd.DataFrame, anchor_a: int, interaction_y: int, pathway: str,
                     resolver: NameResolver, save_path: Optional[str] = None, stage: Optional[str] = None):
    """Bar plot of Δ mean with bootstrap CIs."""
    if dist_df.empty: return
    d_means, ci_los, ci_his = [], [], []
    for m in METRICS:
        a = dist_df.loc[dist_df["group"]=="interacting", m].dropna().to_numpy()
        b = dist_df.loc[dist_df["group"]=="non-interacting", m].dropna().to_numpy()
        dm = float(np.nanmean(a) - np.nanmean(b)) if (len(a) and len(b)) else np.nan
        lo, hi = _bootstrap_mean_diff(a, b, n_boot=BOOT_N, seed=RNG_SEED)
        d_means.append(dm); ci_los.append(lo); ci_his.append(hi)
    x = np.arange(len(METRICS))
    plt.figure(figsize=(8,4.5))
    yerr = [np.array(d_means) - np.array(ci_los), np.array(ci_his) - np.array(d_means)]
    plt.bar(x, d_means, yerr=yerr, capsize=4, alpha=0.85)
    plt.axhline(0, color="#555", linewidth=1)
    plt.xticks(x, [m.capitalize() for m in METRICS])
    ttl = f"Mean diff (int − non) — A={resolver.drug_label(anchor_a)} | Y={int(interaction_y)}\nPathway={resolver.pathway_label(pathway)}"
    if stage: ttl = f"[{stage}] " + ttl
    plt.title(ttl)
    plt.ylabel("Δ mean")
    plt.tight_layout()
    if SAVE_TOGGLE and save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_volcano_for_metric(wdf: pd.DataFrame, metric: str, out_dir: str):
    """Simple volcano plot per metric (Δ mean vs -log10(q))."""
    ensure_dir(out_dir)
    m = metric
    cols = [f"{m}_d_mean", f"{m}_q"]
    if not all(c in wdf.columns for c in cols):
        return
    df = wdf.copy()
    df["effect"] = df[f"{m}_d_mean"].astype(float)
    df["neglog10q"] = df[f"{m}_q"].astype(float).map(lambda x: (0.0 if (pd.isna(x) or x<=0) else -np.log10(x)))
    df = df.sort_values(["neglog10q","effect"], ascending=[False, False]).reset_index(drop=True)
    plt.figure(figsize=(7.5,5.5))
    plt.scatter(df["effect"], df["neglog10q"], s=16, alpha=0.75)
    plt.axhline(-np.log10(0.05), linestyle="--", linewidth=1.0)
    plt.axvline(0.0, linestyle=":", linewidth=1.0)
    plt.xlabel(f"{m.capitalize()} Δ mean (int − non)")
    plt.ylabel("−log10(q)")
    plt.title(f"Volcano — {m} (plain Wilcoxon)")
    fpath = os.path.join(out_dir, f"volcano_{m}.png")
    if SAVE_TOGGLE:
        plt.savefig(fpath, dpi=300, bbox_inches='tight')
    plt.close()

# =========================
# Part 10. Pipelines
# =========================
def run_rank(ddi, anchors_df, subg_base) -> pd.DataFrame:
    """Rank candidates for all anchors."""
    rows = []
    for r in tqdm(list(anchors_df.itertuples(index=False)), total=len(anchors_df), desc="rank"):
        A = int(r.drug); y = int(r.interaction)
        part = evaluate_anchor_label_fast(ddi, A, y, subg_base,
                                          min_group=MIN_GROUP_SIZE,
                                          top_k_pathways=TOPK_PER_ANCHOR,
                                          non_ratio=NON_RATIO,
                                          min_share_ratio=MIN_SHARE_RATIO)
        if not part.empty:
            rows.append(part)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

def run_rank_ext(ddi, anchors_df, subg_base) -> pd.DataFrame:
    """Extended ranking for all anchors."""
    rows = []
    for r in tqdm(list(anchors_df.itertuples(index=False)), total=len(anchors_df), desc="rank_ext"):
        A = int(r.drug); y = int(r.interaction)
        part = evaluate_anchor_label_fast_ext(ddi, A, y, subg_base,
                                              min_group=MIN_GROUP_SIZE,
                                              top_k_pathways=TOPK_PER_ANCHOR,
                                              non_ratio=NON_RATIO,
                                              min_share_ratio=MIN_SHARE_RATIO)
        if not part.empty:
            rows.append(part)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

def rebuild_dist_df_for_combo(A: int, y: int, X: str, ddi: pd.DataFrame, base_dir: str) -> Optional[pd.DataFrame]:
    """Rebuild distribution table for (A,Y,X)."""
    B_int, B_non = B_sets_for_label(ddi, int(A), int(y))
    B_int_shared = [int(b) for b in B_int if drug_has_pathway(int(b), base_dir, X)]
    B_non_shared = [int(b) for b in B_non if drug_has_pathway(int(b), base_dir, X)]
    if not B_int_shared or not B_non_shared:
        return None
    df_i = measure_X_over_B_fast(int(A), X, B_int_shared, base_dir, require_shared=True)
    df_n = measure_X_over_B_fast(int(A), X, B_non_shared, base_dir, require_shared=True)
    if df_i.empty or df_n.empty:
        return None
    df_i["group"] = "interacting"; df_n["group"] = "non-interacting"
    return pd.concat([df_i, df_n], ignore_index=True)

def pick_representative_B_for_graph(A: int, y: int, X: str,
                                    ddi: pd.DataFrame, base_dir: str
                                   ) -> Optional[int]:
    """Pick B in B_int with highest eigenvector centrality at pathway node."""
    B_int, _ = B_sets_for_label(ddi, int(A), int(y))
    B_int_shared = [int(b) for b in B_int if drug_has_pathway(int(b), base_dir, X)]
    if not B_int_shared:
        return None
    Ae, _ = load_graph_files_cached(A, base_dir)
    best_b, best_eig = None, -np.inf
    for b in B_int_shared:
        Be, _ = load_graph_files_cached(int(b), base_dir)
        if Be is None: continue
        G = merged_graph_cached(Ae, Be, int(A), int(b))
        cents = centralities_for_all_nodes_cached(G, int(A), int(b))
        eig = float(cents["eigenvector"].get(X, 0.0))
        if eig > best_eig:
            best_eig = eig; best_b = int(b)
    return best_b if best_b is not None else (int(B_int_shared[0]) if B_int_shared else None)

def visualize_primary_pairs(primary_df: pd.DataFrame, ddi: pd.DataFrame, subg_base: str,
                            resolver: NameResolver, out_viz_stage_dir: str, stage_tag: str):
    """KDE (4 & 2 metrics), mean-shift, and graph per primary combo."""
    if primary_df.empty: return
    for _, row in primary_df.iterrows():
        A = int(row["anchor"]); y = int(row["interaction"]); X = str(row["pathway"])
        anchor_dir = os.path.join(out_viz_stage_dir, f"A{A}"); ensure_dir(anchor_dir)
        niceX = safe_fname(short_label(X, 60))
        dist_df = rebuild_dist_df_for_combo(A, y, X, ddi, subg_base)
        if dist_df is None: continue

        # KDE (4 metrics)
        kde_path = os.path.join(anchor_dir, f"KDE_A{A}_Y{y}_X_{niceX}.png")
        kde_with_top_markers(dist_df, X, A, y, resolver, save_path=kde_path, stage=stage_tag)
        # KDE (2 metrics)
        kde2_path = os.path.join(anchor_dir, f"KDE2_A{A}_Y{y}_X_{niceX}.png")
        kde_with_top_markers(dist_df, X, A, y, resolver, save_path=kde2_path, stage=stage_tag,
                             metrics=("degree","eigenvector"))
        # Mean shift
        meanshift_path = os.path.join(anchor_dir, f"MEANSHIFT_A{A}_Y{y}_X_{niceX}.png")
        mean_shift_panel(dist_df, A, y, X, resolver, save_path=meanshift_path, stage=stage_tag)
        # Representative graph
        b = pick_representative_B_for_graph(A, y, X, ddi, subg_base)
        if b is not None:
            graph_path = os.path.join(anchor_dir, f"Graph_A{A}_Y{y}_B{int(b)}_X_{niceX}.png")
            draw_merged_graph(A, int(b), subg_base, save_path=graph_path,
                              resolver=resolver, highlight_pathway=X,
                              size_by_eig=False, centrality_box=True, interaction_y=y)

def visualize_simple_wilcoxon(wdf: pd.DataFrame, ddi: pd.DataFrame, subg_base: str,
                              resolver: NameResolver, figs_root: str):
    """Per-metric volcano plots and top panels for plain MWU."""
    if wdf.empty: return
    volcano_dir = os.path.join(figs_root, "volcano"); ensure_dir(volcano_dir)
    for m in METRICS:
        plot_volcano_for_metric(wdf, m, volcano_dir)
    top_df = (wdf.sort_values(["min_q","n_p_lt_0_05"], ascending=[True, False]).head(20)).copy()
    panels_root = os.path.join(figs_root, "panels"); ensure_dir(panels_root)
    for _, r in top_df.iterrows():
        A = int(r["anchor"]); y = int(r["interaction"]); X = str(r["pathway"])
        combo_dir = os.path.join(panels_root, f"A{A}_Y{y}"); ensure_dir(combo_dir)
        dist_df = rebuild_dist_df_for_combo(A, y, X, ddi, subg_base)
        if dist_df is None: continue
        niceX = safe_fname(short_label(X, 60))
        kde_path = os.path.join(combo_dir, f"KDE_A{A}_Y{y}_X_{niceX}.png")
        kde_with_top_markers(dist_df, X, A, y, resolver, save_path=kde_path, stage="simple")
        kde2_path = os.path.join(combo_dir, f"KDE2_A{A}_Y{y}_X_{niceX}.png")
        kde_with_top_markers(dist_df, X, A, y, resolver, save_path=kde2_path, stage="simple",
                             metrics=("degree","eigenvector"))
        meanshift_path = os.path.join(combo_dir, f"MEANSHIFT_A{A}_Y{y}_X_{niceX}.png")
        mean_shift_panel(dist_df, A, y, X, resolver, save_path=meanshift_path, stage="simple")
        b = pick_representative_B_for_graph(A, y, X, ddi, subg_base)
        if b is not None:
            graph_path = os.path.join(combo_dir, f"Graph_A{A}_Y{y}_B{int(b)}_X_{niceX}.png")
            draw_merged_graph(A, int(b), subg_base, save_path=graph_path,
                              resolver=resolver, highlight_pathway=X,
                              size_by_eig=False, centrality_box=True, interaction_y=y)

# =========================
# Part 11. Main Entrypoint
# =========================
def main(cli_args=None) -> int:
    parser = argparse.ArgumentParser(
        description="CS_analysis — portable DDI subgraph analysis (save-only)."
    )
    # Primary inputs (all optional thanks to resolve_paths)
    parser.add_argument("--ddi", default="", help="Path to DDI TSV (3 cols: drug1, drug2, interaction)")
    parser.add_argument("--subg_base", default="", help="Folder containing subG_modify/{edges,nodes}")
    parser.add_argument("--save_root", default="", help="Root folder to save all outputs")
    # Optional name-mapping resources
    parser.add_argument("--drkg_nodes", default="", help="DRKG nodes.tsv (optional)")
    parser.add_argument("--hetionet_nodes", default="", help="Hetionet nodes.tsv (optional)")
    # Review-friendly helpers
    parser.add_argument("--review_root", default="", help="Root folder with standard layout (data/, subG_modify/, drkg/, hetionet/)")
    parser.add_argument("--config", default="", help="YAML file with {ddi, subg_base, save_root, drkg_nodes, hetionet_nodes} or paths:{...}")

    args = parser.parse_args(cli_args)

    # Resolve all paths with robust priority
    P = resolve_paths(args)

    # Load data
    ddi = read_ddi(P["ddi"]); _assert_ddi_types(ddi)
    resolver = NameResolver(drkg_nodes_path=P["drkg_nodes"], hetionet_nodes_path=P["hetionet_nodes"])
    anchor_df = calculate_portion(ddi)
    out_paths = OutPaths(P["save_root"])

    # ---- Case 1: filtered_wilcoxon (rank → screen → viz)
    rank_all = run_rank(ddi, anchor_df, P["subg_base"])
    if not rank_all.empty and SAVE_TOGGLE:
        p = os.path.join(out_paths.case_root(CASE_FILTERED), "rank_all.csv")
        (rank_all.assign(anchor=lambda d: d["anchor"].astype(int),
                         interaction=lambda d: d["interaction"].astype(int))
                 .sort_values("final_score", ascending=False)).to_csv(p, index=False)

    if not rank_all.empty:
        primary_pairs, screened_pairs = identify_signal_pairs(
            rank_all, ddi, P["subg_base"],
            delta_thr=CLIFFS_DELTA_THR, fdr_thr=FDR_THR,
            min_each_group=MIN_GROUP_SIZE, min_coverage=MIN_SHARE_RATIO,
            require_two_metrics=KEY_METRICS, top_n_primary=TOP_N_MAIN,
            use_or_on_keys=False, min_n_sig_metrics=2,
            mwu_alternative=MWU_ALTERNATIVE, enable_fallback_when_empty=True
        )
    else:
        primary_pairs = pd.DataFrame(); screened_pairs = pd.DataFrame()

    if SAVE_TOGGLE:
        if not screened_pairs.empty:
            sd = out_paths.stats_dir(CASE_FILTERED, stage=str(screened_pairs.iloc[0].get("selection_stage","strict")))
            screened_pairs.to_csv(os.path.join(sd, "signal_pairs_FDR.csv"), index=False)
        if not primary_pairs.empty:
            sd = out_paths.stats_dir(CASE_FILTERED, stage=str(primary_pairs.iloc[0].get("selection_stage","strict")))
            primary_pairs.to_csv(os.path.join(sd, "primary_signals.csv"), index=False)

    if not primary_pairs.empty:
        stage = str(primary_pairs.iloc[0].get("selection_stage","strict"))
        viz_stage_dir = out_paths.viz_dir(CASE_FILTERED, stage=stage)
        visualize_primary_pairs(primary_df=primary_pairs, ddi=ddi, subg_base=P["subg_base"],
                                resolver=resolver, out_viz_stage_dir=viz_stage_dir, stage_tag=stage)

    # ---- Case 2: simple_wilcoxon (plain MWU)
    if not rank_all.empty:
        wdf = run_simple_wilcoxon(rank_all, ddi, P["subg_base"], min_each_group=MIN_GROUP_SIZE,
                                  mwu_alternative=MWU_ALTERNATIVE)
        if SAVE_TOGGLE and not wdf.empty:
            sd = out_paths.stats_dir(CASE_SIMPLE)
            wdf.to_csv(os.path.join(sd, "wilcoxon_results.csv"), index=False)
            figs_root = out_paths.figs_dir(CASE_SIMPLE)
            visualize_simple_wilcoxon(wdf, ddi, P["subg_base"], resolver, figs_root)

    # ---- Case 3: extended (rank_ext → screen_ext → viz)
    rank_all_ext = run_rank_ext(ddi, anchor_df, P["subg_base"])
    if not rank_all_ext.empty and SAVE_TOGGLE:
        p = os.path.join(out_paths.case_root(CASE_EXTENDED), "rank_all_ext.csv")
        (rank_all_ext.assign(anchor=lambda d: d["anchor"].astype(int),
                             interaction=lambda d: d["interaction"].astype(int))
                     .sort_values("final_score", ascending=False)).to_csv(p, index=False)

    if not rank_all_ext.empty:
        primary_pairs_ext, screened_pairs_ext = identify_signal_pairs_ext(
            rank_all_ext, ddi, P["subg_base"], metrics=METRICS_EXT,
            delta_thr=CLIFFS_DELTA_THR, fdr_thr=FDR_THR, min_each_group=MIN_GROUP_SIZE,
            min_coverage=MIN_SHARE_RATIO, require_two_metrics=KEY_METRICS,
            top_n_primary=TOP_N_MAIN, use_or_on_keys=False, min_n_sig_metrics=2,
            mwu_alternative=MWU_ALTERNATIVE, enable_fallback_when_empty=True
        )
    else:
        primary_pairs_ext = pd.DataFrame(); screened_pairs_ext = pd.DataFrame()

    if SAVE_TOGGLE:
        if not screened_pairs_ext.empty:
            sd = out_paths.stats_dir(CASE_EXTENDED, stage=str(screened_pairs_ext.iloc[0].get("selection_stage","strict")))
            screened_pairs_ext.to_csv(os.path.join(sd, "signal_candidates_ext.csv"), index=False)
        if not primary_pairs_ext.empty:
            sd = out_paths.stats_dir(CASE_EXTENDED, stage=str(primary_pairs_ext.iloc[0].get("selection_stage","strict")))
            primary_pairs_ext.to_csv(os.path.join(sd, "primary_signals_ext.csv"), index=False)

    if not primary_pairs_ext.empty:
        stage = str(primary_pairs_ext.iloc[0].get("selection_stage","strict"))
        viz_stage_dir = out_paths.viz_dir(CASE_EXTENDED, stage=stage)
        visualize_primary_pairs(primary_df=primary_pairs_ext, ddi=ddi, subg_base=P["subg_base"],
                                resolver=resolver, out_viz_stage_dir=viz_stage_dir, stage_tag=f"extended-{stage}")

    return 0

# =========================
# Part 12. CLI
# =========================
if __name__ == "__main__":
    sys.exit(main())