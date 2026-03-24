from __future__ import annotations

import logging
import math
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from utils.common import set_seeds

logger = logging.getLogger("TRACE-DDI")


def canonical_pair(d1: str, d2: str) -> Tuple[str, str]:
    # Canonicalize unordered pair
    return (d1, d2) if d1 <= d2 else (d2, d1)


def canonicalize_df_pairs(
    df: pd.DataFrame,
    col1: str = "drug1",
    col2: str = "drug2",
) -> pd.DataFrame:
    # Canonicalize pair columns
    a = df[col1].astype(str).to_numpy()
    b = df[col2].astype(str).to_numpy()
    mask = a <= b
    lo = np.where(mask, a, b)
    hi = np.where(mask, b, a)

    out = df.copy()
    out[col1] = lo
    out[col2] = hi
    return out


def dedup_pairs(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    # Drop exact duplicates
    before = len(df)
    out = df.drop_duplicates(subset=list(cols), keep="first").reset_index(drop=True)
    after = len(out)
    logger.info(
        "Dropped %d duplicates (%.3f%%).",
        before - after,
        100.0 * (before - after) / max(before, 1),
    )
    return out


def make_drug_holdout_split_strict(
    df: pd.DataFrame,
    holdout_frac: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], int]:
    """
    Strict drug holdout:
    - Train: both drugs are in seen set
    - Test: both drugs are in holdout set
    - Cross pairs dropped
    """
    drugs = sorted(set(df["drug1"].astype(str)).union(set(df["drug2"].astype(str))))
    rng = np.random.default_rng(seed)
    n_hold = max(1, int(len(drugs) * holdout_frac))
    holdout_set = set(rng.choice(drugs, size=n_hold, replace=False).tolist())
    holdout_list: List[str] = sorted(list(holdout_set))

    d1 = df["drug1"].astype(str)
    d2 = df["drug2"].astype(str)

    in_hold = d1.isin(holdout_list) & d2.isin(holdout_list)
    in_seen = (~d1.isin(holdout_list)) & (~d2.isin(holdout_list))

    test_df = df.loc[in_hold].copy().reset_index(drop=True)
    train_df = df.loc[in_seen].copy().reset_index(drop=True)
    dropped = int((~(in_hold | in_seen)).sum())

    logger.info(
        "[drug_holdout] drugs=%d holdout=%d (%.1f%%) train_pairs=%d test_pairs=%d dropped_cross=%d",
        len(drugs),
        len(holdout_set),
        100.0 * len(holdout_set) / max(len(drugs), 1),
        len(train_df),
        len(test_df),
        dropped,
    )
    return train_df, test_df, holdout_list, dropped


def encode_multiclass_labels(
    df: pd.DataFrame,
    col: str = "interaction",
) -> Tuple[pd.DataFrame, int]:
    s = df[col].astype("category")
    codes = s.cat.codes.astype(int)

    out = df.copy()
    out[col] = codes
    n_classes = int(pd.Series(codes).nunique())
    return out, n_classes


def compute_drug_overlap_stats(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> Dict[str, int]:
    tr_drugs = set(train_df["drug1"].astype(str)).union(
        set(train_df["drug2"].astype(str))
    )
    va_drugs = set(val_df["drug1"].astype(str)).union(set(val_df["drug2"].astype(str)))
    shared = tr_drugs.intersection(va_drugs)
    return {
        "n_train_drugs": len(tr_drugs),
        "n_val_drugs": len(va_drugs),
        "n_shared_drugs": len(shared),
    }


def build_binary_dataset(
    pos_pairs_path: str,
    smiles_path: str,
    vec_path: str,
    out_path: str,
    neg_ratio: float,
    seed: int,
    pair_dedup: bool,
) -> None:
    """
    Positive = pairs present in pos_pairs_path.
    Negative = sampled pairs not in positives.
    Candidate drugs = intersection of SMILES drugs and vector drugs.
    """
    set_seeds(seed)

    pos_raw = pd.read_csv(pos_pairs_path, sep="\t", header=None)
    if pos_raw.shape[1] < 2:
        raise ValueError("pos_pairs_path must have at least 2 columns: drug1, drug2.")

    pos_raw = pos_raw.iloc[:, :2].copy()
    pos_raw.columns = pd.Index(["drug1", "drug2"])
    pos_raw["drug1"] = pos_raw["drug1"].astype(str)
    pos_raw["drug2"] = pos_raw["drug2"].astype(str)

    if pair_dedup:
        pos_raw = canonicalize_df_pairs(pos_raw)
        pos_raw = dedup_pairs(pos_raw, ["drug1", "drug2"])

    pos_set: Set[Tuple[str, str]] = set()
    for a, b in zip(pos_raw["drug1"].tolist(), pos_raw["drug2"].tolist()):
        pos_set.add(canonical_pair(a, b) if pair_dedup else (a, b))

    smiles_df = pd.read_csv(
        smiles_path,
        sep="\t",
        header=None,
        names=["drug", "smiles"],
    )
    smiles_df["drug"] = smiles_df["drug"].astype(str)
    smiles_drugs: Set[str] = set(smiles_df["drug"].tolist())

    vec_df = pd.read_csv(vec_path, index_col=0)
    vec_drugs: Set[str] = set(vec_df.index.astype(str).tolist())

    drugs = sorted(list(smiles_drugs.intersection(vec_drugs)))
    if len(drugs) < 2:
        raise RuntimeError("Not enough drugs after intersecting smiles and vectors.")

    pos_filtered: List[Tuple[str, str]] = []
    for a, b in pos_set:
        if (
            a in smiles_drugs
            and b in smiles_drugs
            and a in vec_drugs
            and b in vec_drugs
        ):
            pos_filtered.append((a, b))
    pos_set = set(pos_filtered)

    n_pos = len(pos_set)
    if n_pos == 0:
        raise RuntimeError(
            "No valid positive pairs after filtering by available drugs."
        )

    n_neg = int(math.ceil(n_pos * neg_ratio))
    logger.info(
        "Binary build: positives=%d, target_negatives=%d (ratio=%.3f)",
        n_pos,
        n_neg,
        neg_ratio,
    )

    rng = np.random.default_rng(seed)
    neg_set: Set[Tuple[str, str]] = set()
    max_tries = n_neg * 50
    tries = 0

    while len(neg_set) < n_neg and tries < max_tries:
        tries += 1
        a = drugs[int(rng.integers(0, len(drugs)))]
        b = drugs[int(rng.integers(0, len(drugs)))]

        if a == b:
            continue

        p = canonical_pair(a, b) if pair_dedup else (a, b)
        if p in pos_set or p in neg_set:
            continue
        neg_set.add(p)

    if len(neg_set) < n_neg:
        logger.warning(
            "Negative sampling hit limit: got %d / %d negatives.",
            len(neg_set),
            n_neg,
        )

    out_rows: List[Tuple[str, str, int]] = []
    out_rows.extend([(a, b, 1) for (a, b) in pos_set])
    out_rows.extend([(a, b, 0) for (a, b) in neg_set])

    out_df = (
        pd.DataFrame(
            out_rows,
            columns=pd.Index(["drug1", "drug2", "interaction"]),
        )
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )

    out_df.to_csv(out_path, sep="\t", index=False, header=False)
    logger.info("Wrote binary dataset to %s (n=%d).", out_path, len(out_df))
