from __future__ import annotations

from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd

from cs_analysis_utils.config import MIN_SHARE_RATIO
from cs_analysis_utils.graph_io import get_pathway_set_cached, load_graph_files_cached


def _to_int_scalar(x: Any) -> int:
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return int(x)
    if isinstance(x, str):
        return int(x)
    raise TypeError(f"Cannot convert {type(x).__name__} to int")


def _to_float_scalar(x: Any) -> float:
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    if isinstance(x, str):
        return float(x)
    raise TypeError(f"Cannot convert {type(x).__name__} to float")


def partners_of(ddi: pd.DataFrame, drug_id: int) -> pd.DataFrame:
    drug_id_i = int(drug_id)
    mask = (ddi["drug1"] == drug_id_i) | (ddi["drug2"] == drug_id_i)
    out = ddi.loc[mask].copy()
    if not isinstance(out, pd.DataFrame):
        raise TypeError("Expected DataFrame from partners_of")
    return out


def calculate_portion(ddi_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, int | float]] = []

    for interaction, group in ddi_df.groupby("interaction", sort=False):
        if not isinstance(group, pd.DataFrame):
            raise TypeError("Expected grouped object to be DataFrame")

        drug1_counts = group["drug1"].value_counts()
        drug2_counts = group["drug2"].value_counts()
        counts = drug1_counts.add(drug2_counts, fill_value=0)

        portions = (counts / (len(group) * 2)) * 100.0
        selected = portions[portions >= 50.0]

        for drug_key, portion_val in selected.items():
            rows.append(
                {
                    "interaction": _to_int_scalar(interaction),
                    "drug": _to_int_scalar(drug_key),
                    "portion": _to_float_scalar(portion_val),
                }
            )

    df = pd.DataFrame(rows)

    if df.empty:
        return pd.DataFrame(
            {
                "interaction": pd.Series(dtype="int64"),
                "drug": pd.Series(dtype="int64"),
                "portion": pd.Series(dtype="float64"),
            }
        )

    out = df[["interaction", "drug", "portion"]]
    if not isinstance(out, pd.DataFrame):
        raise TypeError("Expected DataFrame from calculate_portion")
    return out


def B_sets_for_label(
    ddi: pd.DataFrame,
    anchor_a: int,
    label_y: int,
) -> Tuple[List[int], List[int]]:
    anchor = int(anchor_a)
    label = int(label_y)

    ab = partners_of(ddi, anchor)
    mask = ((ab["drug1"] == anchor) & (ab["interaction"] == label)) | (
        (ab["drug2"] == anchor) & (ab["interaction"] == label)
    )

    pair_values = ab.loc[mask, ["drug1", "drug2"]].to_numpy().ravel()

    Bs_y: Set[int] = set()
    for value in pair_values:
        Bs_y.add(_to_int_scalar(value))
    Bs_y.discard(anchor)

    all_drugs: Set[int] = set()
    for value in ddi["drug1"].tolist():
        all_drugs.add(_to_int_scalar(value))
    for value in ddi["drug2"].tolist():
        all_drugs.add(_to_int_scalar(value))

    return sorted(Bs_y), sorted(all_drugs - Bs_y - {anchor})


def drug_has_pathway(drug_id: int, base_dir: str, pathway: str) -> bool:
    return pathway in get_pathway_set_cached(int(drug_id), base_dir)


def pathway_candidates_shared(
    anchor_a: int,
    label_y: int,
    base_dir: str,
    ddi: pd.DataFrame,
    min_share_ratio: float = MIN_SHARE_RATIO,
    max_expand_B: int = 200,
) -> Tuple[Set[str], Dict[str, Tuple[int, float]]]:
    _, anchor_nodes = load_graph_files_cached(int(anchor_a), base_dir)
    if anchor_nodes is None:
        return set(), {}

    A_paths = get_pathway_set_cached(int(anchor_a), base_dir)
    B_int, _ = B_sets_for_label(ddi, int(anchor_a), int(label_y))
    if not B_int:
        return set(), {}

    pool = B_int[:max_expand_B]
    cnt: Dict[str, int] = {}

    for b in pool:
        shared = A_paths & get_pathway_set_cached(int(b), base_dir)
        for pathway in shared:
            cnt[pathway] = cnt.get(pathway, 0) + 1

    cand: Set[str] = set()
    cov: Dict[str, Tuple[int, float]] = {}
    nB = max(1, len(pool))

    for pathway, count in cnt.items():
        ratio = count / nB
        if ratio >= min_share_ratio:
            cand.add(pathway)
            cov[pathway] = (count, ratio)

    return cand, cov
