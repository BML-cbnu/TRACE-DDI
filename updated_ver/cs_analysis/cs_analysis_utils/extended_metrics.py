from typing import Dict, Iterable, Mapping, cast

import networkx as nx
import numpy as np
import pandas as pd

from cs_analysis_utils.config import METRICS_EXT, REP_W_EXT


def radiality_per_component(
    G: nx.Graph,
    nodes: Iterable[str],
    dist_map: Dict[str, Dict[str, int]],
) -> Dict[str, float]:
    sub = G.subgraph(nodes)
    if sub.number_of_nodes() <= 1:
        return {str(n): 1.0 for n in sub.nodes()}

    max_d = 1
    for v in sub.nodes():
        v_key = str(v)
        for u, d in dist_map[v_key].items():
            if u in sub and u != v_key and d > max_d:
                max_d = d

    D = max(1, max_d)
    out: Dict[str, float] = {}

    for v in sub.nodes():
        v_key = str(v)
        s = 0.0
        cnt = 0
        for u in sub.nodes():
            u_key = str(u)
            if u_key == v_key:
                continue
            d = dist_map[v_key].get(u_key)
            if d is None:
                continue
            s += D + 1 - d
            cnt += 1
        out[v_key] = (s / max(1, cnt)) / D

    return out


def compute_extra_metrics(G: nx.Graph) -> Dict[str, Dict[str, float]]:
    clustering_result = nx.clustering(G)
    eps = 1e-9

    if isinstance(clustering_result, dict):
        clustering_map: Mapping[object, float] = cast(
            Mapping[object, float], clustering_result
        )
    else:
        clustering_map = {str(n): float(clustering_result) for n in G.nodes()}

    clustering_inv: Dict[str, float] = {
        str(n): 1.0 / max(eps, float(v)) for n, v in clustering_map.items()
    }

    ecc: Dict[str, int] = {}
    for comp in nx.connected_components(G):
        sub = G.subgraph(comp)
        try:
            sub_ecc_raw = nx.eccentricity(sub)
            if isinstance(sub_ecc_raw, dict):
                normalized_sub_ecc: Dict[str, int] = {
                    str(k): int(v) for k, v in sub_ecc_raw.items()
                }
            else:
                normalized_sub_ecc = {str(n): int(sub_ecc_raw) for n in sub.nodes()}
        except Exception:
            spl = dict(nx.all_pairs_shortest_path_length(sub))
            normalized_sub_ecc = {
                str(v): int(max(d.values()) if len(d) else 0) for v, d in spl.items()
            }
        ecc.update(normalized_sub_ecc)

    ecc_inv: Dict[str, float] = {
        str(n): (1.0 / float(v) if v > 0 else 0.0) for n, v in ecc.items()
    }

    all_spl_raw = dict(nx.all_pairs_shortest_path_length(G))
    all_spl: Dict[str, Dict[str, int]] = {
        str(src): {str(dst): int(dist) for dst, dist in dist_map.items()}
        for src, dist_map in all_spl_raw.items()
    }

    rad: Dict[str, float] = {}
    for comp in nx.connected_components(G):
        comp_nodes = [str(n) for n in comp]
        rad.update(radiality_per_component(G, comp_nodes, all_spl))

    return {
        "clustering_inv": clustering_inv,
        "eccentricity_inv": ecc_inv,
        "radiality": rad,
    }


def separation_score_ext(
    dist_df: pd.DataFrame,
    weights: Dict[str, float] = REP_W_EXT,
    metrics: tuple[str, ...] = METRICS_EXT,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    gi = dist_df["group"] == "interacting"
    gn = dist_df["group"] == "non-interacting"

    for m in metrics:
        med_i = float(np.nanmedian(dist_df.loc[gi, m])) if bool(gi.any()) else 0.0
        med_n = float(np.nanmedian(dist_df.loc[gn, m])) if bool(gn.any()) else 0.0
        out[f"diff_med_{m}"] = med_i - med_n

    out["sep_score"] = sum(
        weights.get(k, 0.0) * out.get(f"diff_med_{k}", 0.0)
        for k in metrics
        if k in weights
    )
    return out
