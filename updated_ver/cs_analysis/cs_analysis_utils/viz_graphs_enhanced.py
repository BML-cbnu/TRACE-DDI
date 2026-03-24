"""
viz_graphs_enhanced.py
"""

from __future__ import annotations

import math
import os
from typing import Dict, Iterable, Optional, Protocol, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from cs_analysis_utils.config import SAVE_TOGGLE
from cs_analysis_utils.graph_io import (
    centralities_for_all_nodes_cached,
    load_graph_files_cached,
    merged_graph_cached,
)
from cs_analysis_utils.io_utils import ensure_dir, short_label


class ResolverProtocol(Protocol):
    def drug_label(self, x: int) -> str: ...
    def pathway_label(self, node_info: object) -> str: ...


def _to_optional_int_scalar(x: object) -> Optional[int]:
    if x is None:
        return None

    try:
        na_check = pd.isna(x)
        if isinstance(na_check, (bool, np.bool_)) and bool(na_check):
            return None
    except Exception:
        pass

    if isinstance(x, (int, np.integer)):
        return int(x)

    if isinstance(x, (float, np.floating)):
        fx = float(x)
        if math.isnan(fx):
            return None
        return int(fx)

    if isinstance(x, str):
        s = x.strip()
        if s == "":
            return None
        try:
            return int(float(s))
        except ValueError:
            return None

    try:
        return int(float(x))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def node_name_to_nodeid_pref_node_id(
    nodes: Optional[pd.DataFrame],
) -> Dict[str, int]:
    if nodes is None or nodes.empty:
        return {}

    col: Optional[str] = None
    if "node_id" in nodes.columns:
        col = "node_id"
    elif "ori_node_id" in nodes.columns:
        col = "ori_node_id"

    if col is None:
        return {}

    out: Dict[str, int] = {}
    for _, row in nodes.iterrows():
        node_id = _to_optional_int_scalar(row[col])
        if node_id is None:
            continue
        out[str(row["node_name"])] = node_id
    return out


def find_drug_node_name(
    drug_id: int,
    nodes: Optional[pd.DataFrame],
) -> Optional[str]:
    if nodes is None or nodes.empty:
        return None

    for id_col in ("node_id", "ori_node_id"):
        if id_col not in nodes.columns:
            continue

        for _, row in nodes.iterrows():
            node_val = _to_optional_int_scalar(row[id_col])
            if node_val is None:
                continue
            if node_val == int(drug_id):
                return str(row["node_name"])

    return None


def build_node_color_map_enhanced(
    G: nx.Graph,
    An: Optional[pd.DataFrame],
    Bn: Optional[pd.DataFrame],
    drug_a: int,
    drug_b: int,
    highlight: Optional[Iterable[str]] = None,
) -> Dict[str, str]:
    highlight_set = set(highlight or [])
    mapA = node_name_to_nodeid_pref_node_id(An)
    mapB = node_name_to_nodeid_pref_node_id(Bn)

    colors: Dict[str, str] = {}
    for n in G.nodes():
        ns = str(n)

        if ns in highlight_set:
            colors[ns] = "#FFD54F"
            continue

        if (ns in mapA and mapA[ns] == int(drug_a)) or (
            ns in mapB and mapB[ns] == int(drug_b)
        ):
            colors[ns] = "#D32F2F"
            continue

        inA = ns in mapA
        inB = ns in mapB

        if inA and inB:
            colors[ns] = "#6A1B9A"
        elif inA:
            colors[ns] = "#1E88E5"
        elif inB:
            colors[ns] = "#43A047"
        else:
            colors[ns] = "#B0B0B0"

    return colors


def compute_node_sizes_enhanced(
    G: nx.Graph,
    eig_map: Dict[str, float],
    highlight_pathway: Optional[str] = None,
    drug_a_node: Optional[str] = None,
    drug_b_node: Optional[str] = None,
) -> Dict[str, float]:
    nodes = list(G.nodes())
    eig_vals = np.array([float(eig_map.get(str(n), 0.0)) for n in nodes], dtype=float)

    if eig_vals.size == 0 or float(eig_vals.max()) <= 0.0:
        base_sizes = {str(n): 420.0 for n in nodes}
    else:
        scaled = 260.0 + 740.0 * (eig_vals / float(eig_vals.max()))
        base_sizes = {str(n): float(sz) for n, sz in zip(nodes, scaled)}

    if highlight_pathway is not None and highlight_pathway in base_sizes:
        base_sizes[highlight_pathway] = max(base_sizes[highlight_pathway], 1250.0)

    if drug_a_node is not None and drug_a_node in base_sizes:
        base_sizes[drug_a_node] = max(base_sizes[drug_a_node], 1100.0)

    if drug_b_node is not None and drug_b_node in base_sizes:
        base_sizes[drug_b_node] = max(base_sizes[drug_b_node], 1100.0)

    return base_sizes


def select_labels_by_priority(
    G: nx.Graph,
    eig_map: Dict[str, float],
    max_labels: int = 25,
    always_include: Optional[Iterable[str]] = None,
) -> Dict[str, str]:
    include = set(always_include or [])
    ranked = sorted(
        [str(n) for n in G.nodes()],
        key=lambda x: float(eig_map.get(x, 0.0)),
        reverse=True,
    )

    chosen: list[str] = []
    for n in ranked:
        if len(chosen) >= max_labels:
            break
        chosen.append(n)

    for n in include:
        if n in G and n not in chosen:
            chosen.append(n)

    return {n: short_label(n, 34) for n in chosen}


def compute_anchor_layout(
    G: nx.Graph,
    drug_a_node: Optional[str],
    drug_b_node: Optional[str],
    highlight_pathway: Optional[str],
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    fixed_pos: Dict[str, np.ndarray] = {}
    fixed_nodes: list[str] = []

    if drug_a_node is not None and drug_a_node in G:
        fixed_pos[drug_a_node] = np.array([-3.0, 0.0], dtype=float)
        fixed_nodes.append(drug_a_node)

    if highlight_pathway is not None and highlight_pathway in G:
        fixed_pos[highlight_pathway] = np.array([0.0, 0.0], dtype=float)
        fixed_nodes.append(highlight_pathway)

    if drug_b_node is not None and drug_b_node in G:
        fixed_pos[drug_b_node] = np.array([3.0, 0.0], dtype=float)
        fixed_nodes.append(drug_b_node)

    if not fixed_nodes:
        return nx.spring_layout(G, seed=seed)

    return nx.spring_layout(
        G,
        seed=seed,
        pos=fixed_pos,
        fixed=fixed_nodes,
        k=1.7 / max(np.sqrt(max(G.number_of_nodes(), 1)), 1.0),
        iterations=350,
    )


def _unit_vector(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-12:
        theta = np.random.default_rng(42).uniform(0.0, 2.0 * np.pi)
        return np.array([math.cos(theta), math.sin(theta)], dtype=float)
    return vec / norm


def repel_nodes_from_anchor(
    pos: Dict[str, np.ndarray],
    anchor_node: str,
    min_radius: float,
    skip_nodes: Optional[Iterable[str]] = None,
) -> Dict[str, np.ndarray]:
    if anchor_node not in pos:
        return pos

    skip = set(skip_nodes or [])
    skip.add(anchor_node)

    anchor_xy = pos[anchor_node]

    for node, xy in pos.items():
        if node in skip:
            continue

        diff = xy - anchor_xy
        dist = float(np.linalg.norm(diff))
        if dist < min_radius:
            direction = _unit_vector(diff)
            pos[node] = anchor_xy + direction * min_radius

    return pos


def repel_nodes_from_anchors_variable(
    pos: Dict[str, np.ndarray],
    drug_a_node: Optional[str],
    drug_b_node: Optional[str],
    highlight_pathway: Optional[str],
) -> Dict[str, np.ndarray]:
    anchor_set = {
        n for n in (drug_a_node, drug_b_node, highlight_pathway) if n is not None
    }

    if highlight_pathway is not None:
        pos = repel_nodes_from_anchor(
            pos=pos,
            anchor_node=highlight_pathway,
            min_radius=1.10,
            skip_nodes=anchor_set,
        )

    if drug_a_node is not None:
        pos = repel_nodes_from_anchor(
            pos=pos,
            anchor_node=drug_a_node,
            min_radius=0.95,
            skip_nodes=anchor_set,
        )

    if drug_b_node is not None:
        pos = repel_nodes_from_anchor(
            pos=pos,
            anchor_node=drug_b_node,
            min_radius=0.95,
            skip_nodes=anchor_set,
        )

    return pos


def relax_non_anchor_nodes(
    G: nx.Graph,
    pos: Dict[str, np.ndarray],
    drug_a_node: Optional[str],
    drug_b_node: Optional[str],
    highlight_pathway: Optional[str],
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    fixed_nodes = [
        n
        for n in (drug_a_node, highlight_pathway, drug_b_node)
        if n is not None and n in G and n in pos
    ]

    if not fixed_nodes:
        return pos

    return nx.spring_layout(
        G,
        seed=seed,
        pos=pos,
        fixed=fixed_nodes,
        k=1.35 / max(np.sqrt(max(G.number_of_nodes(), 1)), 1.0),
        iterations=80,
    )


def draw_merged_graph_enhanced(
    drug_a: int,
    drug_b: int,
    base_dir: str,
    save_path: str,
    resolver: ResolverProtocol,
    highlight_pathway: Optional[str] = None,
    interaction_y: Optional[int] = None,
    seed: int = 42,
    figsize: Tuple[int, int] = (13, 9),
    max_labels: int = 25,
) -> None:
    Ae, An = load_graph_files_cached(int(drug_a), base_dir)
    Be, Bn = load_graph_files_cached(int(drug_b), base_dir)
    if Ae is None or Be is None:
        return

    G = merged_graph_cached(Ae, Be, int(drug_a), int(drug_b))
    cents = centralities_for_all_nodes_cached(G, int(drug_a), int(drug_b))
    eig_map = cents["eigenvector"]

    drug_a_node = find_drug_node_name(int(drug_a), An)
    drug_b_node = find_drug_node_name(int(drug_b), Bn)

    colors = build_node_color_map_enhanced(
        G,
        An,
        Bn,
        drug_a,
        drug_b,
        highlight=[highlight_pathway] if highlight_pathway else None,
    )

    node_sizes = compute_node_sizes_enhanced(
        G,
        eig_map,
        highlight_pathway=highlight_pathway,
        drug_a_node=drug_a_node,
        drug_b_node=drug_b_node,
    )

    always_include: list[str] = []
    if highlight_pathway is not None:
        always_include.append(highlight_pathway)
    if drug_a_node is not None:
        always_include.append(drug_a_node)
    if drug_b_node is not None:
        always_include.append(drug_b_node)

    labels = select_labels_by_priority(
        G,
        eig_map,
        max_labels=max_labels,
        always_include=always_include,
    )

    pos = compute_anchor_layout(
        G=G,
        drug_a_node=drug_a_node,
        drug_b_node=drug_b_node,
        highlight_pathway=highlight_pathway,
        seed=seed,
    )

    pos = repel_nodes_from_anchors_variable(
        pos=pos,
        drug_a_node=drug_a_node,
        drug_b_node=drug_b_node,
        highlight_pathway=highlight_pathway,
    )

    pos = relax_non_anchor_nodes(
        G=G,
        pos=pos,
        drug_a_node=drug_a_node,
        drug_b_node=drug_b_node,
        highlight_pathway=highlight_pathway,
        seed=seed,
    )

    pos = repel_nodes_from_anchors_variable(
        pos=pos,
        drug_a_node=drug_a_node,
        drug_b_node=drug_b_node,
        highlight_pathway=highlight_pathway,
    )

    plt.figure(figsize=figsize)

    nx.draw_networkx_edges(
        G,
        pos,
        edge_color=[d.get("color", "#BDBDBD") for _, _, d in G.edges(data=True)],
        width=1.0,
        alpha=0.45,
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=[colors[str(n)] for n in G.nodes()],
        node_size=[node_sizes[str(n)] for n in G.nodes()],
        linewidths=0.8,
        edgecolors="#424242",
    )

    nx.draw_networkx_labels(
        G,
        pos,
        labels=labels,
        font_size=8,
    )

    if highlight_pathway is not None and highlight_pathway in G:
        info = "\n".join(
            [
                f"degree: {cents['degree'].get(highlight_pathway, 0.0):.4f}",
                f"betweenness: {cents['betweenness'].get(highlight_pathway, 0.0):.4f}",
                f"closeness: {cents['closeness'].get(highlight_pathway, 0.0):.4f}",
                f"eigenvector: {cents['eigenvector'].get(highlight_pathway, 0.0):.4f}",
            ]
        )
        plt.text(
            0.98,
            0.98,
            info,
            transform=plt.gca().transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.80, edgecolor="#BDBDBD"),
        )

    from matplotlib.patches import Patch

    legend_elems = [
        Patch(facecolor="#D32F2F", label="Drug (A/B, exact)"),
        Patch(facecolor="#1E88E5", label="A subgraph"),
        Patch(facecolor="#43A047", label="B subgraph"),
        Patch(facecolor="#6A1B9A", label="A∩B"),
        Patch(facecolor="#FFD54F", label="Highlighted pathway"),
    ]
    plt.legend(handles=legend_elems, loc="lower left", framealpha=0.95)

    ttl = (
        f"Enhanced merged — A={resolver.drug_label(drug_a)} "
        f"+ B={resolver.drug_label(drug_b)}"
    )
    if highlight_pathway is not None:
        ttl += f"\n| highlight: {resolver.pathway_label(highlight_pathway)}"
    if interaction_y is not None:
        ttl += f" | Y={int(interaction_y)}"

    plt.title(ttl, pad=18)
    plt.axis("off")
    plt.tight_layout()

    if SAVE_TOGGLE and save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=320, bbox_inches="tight")

    plt.close()
