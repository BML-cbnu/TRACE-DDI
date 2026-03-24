import os
from typing import Dict, Optional, Set, Tuple

import networkx as nx
import pandas as pd

from cs_analysis_utils.config import EIG_MAX_ITER, EIG_TOL, EIGEN_USE_POWER

GraphFiles = Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]

_EDGES_NODES_CACHE: Dict[Tuple[int, str], GraphFiles] = {}
_GRAPH_CACHE: Dict[Tuple[int, int], nx.Graph] = {}
_CENTRALITY_CACHE: Dict[Tuple[int, int], Dict[str, Dict[str, float]]] = {}
_PATHWAY_SET_CACHE: Dict[Tuple[int, str], Set[str]] = {}


def load_graph_files(drug_id: int, base_dir: str) -> GraphFiles:
    efile = os.path.join(base_dir, f"edges/compound{int(drug_id)}_edges.tsv")
    nfile = os.path.join(base_dir, f"nodes/compound{int(drug_id)}_nodes.tsv")

    if not (os.path.exists(efile) and os.path.exists(nfile)):
        return None, None

    edges = pd.read_csv(
        efile,
        sep="\t",
        header=None,
        names=[
            "node1",
            "relation_id",
            "node2",
            "ori_node1_num",
            "ori_node2_num",
            "node1_name",
            "node2_name",
        ],
    )

    try:
        nodes = pd.read_csv(
            nfile,
            sep="\t",
            header=None,
            names=[
                "node_id",
                "node_name",
                "node_type",
                "visited_count",
                "spread_value",
                "ori_node_id",
            ],
        )
    except Exception:
        nodes = pd.read_csv(
            nfile,
            sep="\t",
            header=None,
            names=[
                "node_id",
                "node_name",
                "node_type",
                "visited_count",
                "spread_value",
                "ori_nude_id",
            ],
        )
        nodes = nodes.rename(columns={"ori_nude_id": "ori_node_id"})

    return edges, nodes


def _std_nodes(nodes: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if nodes is None:
        return None
    out = nodes.copy()
    out["node_name"] = out["node_name"].astype(str).str.strip()
    return out


def load_graph_files_cached(drug_id: int, base_dir: str) -> GraphFiles:
    key = (int(drug_id), base_dir)
    if key in _EDGES_NODES_CACHE:
        return _EDGES_NODES_CACHE[key]

    e, n = load_graph_files(int(drug_id), base_dir)
    n = _std_nodes(n)
    _EDGES_NODES_CACHE[key] = (e, n)
    return e, n


def get_pathway_set_cached(drug_id: int, base_dir: str) -> Set[str]:
    key = (int(drug_id), base_dir)
    if key in _PATHWAY_SET_CACHE:
        return _PATHWAY_SET_CACHE[key]

    _, n = load_graph_files_cached(int(drug_id), base_dir)
    if n is None:
        s: Set[str] = set()
    else:
        s = set(
            n.loc[
                (n["node_type"] == 9)
                | (n["node_name"].astype(str).str.startswith("Pathway::")),
                "node_name",
            ].astype(str)
        )

    _PATHWAY_SET_CACHE[key] = s
    return s


def merged_graph_edges(
    Ae: Optional[pd.DataFrame], Be: Optional[pd.DataFrame]
) -> nx.Graph:
    G = nx.Graph()

    def _add(edges: Optional[pd.DataFrame], color: str) -> None:
        if edges is None:
            return
        for _, r in edges.iterrows():
            u, v = str(r["node1_name"]), str(r["node2_name"])
            G.add_edge(u, v, relation=r["relation_id"], color=color)

    _add(Ae, "#1E88E5")
    _add(Be, "#43A047")
    return G


def merged_graph_cached(Ae, Be, A_id: int, B_id: int) -> nx.Graph:
    key = (int(A_id), int(B_id))
    if key in _GRAPH_CACHE:
        return _GRAPH_CACHE[key]

    G = merged_graph_edges(Ae, Be)
    _GRAPH_CACHE[key] = G
    return G


def centralities_for_all_nodes_cached(
    G: nx.Graph,
    A_id: int,
    B_id: int,
) -> Dict[str, Dict[str, float]]:
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

    out = {
        "degree": deg,
        "betweenness": btw,
        "closeness": clo,
        "eigenvector": eig,
    }
    _CENTRALITY_CACHE[key] = out
    return out
