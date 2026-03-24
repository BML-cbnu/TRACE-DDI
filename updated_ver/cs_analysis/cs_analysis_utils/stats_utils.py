from __future__ import annotations

from typing import Callable, Dict, List, Tuple, cast

import numpy as np
import pandas as pd

from cs_analysis_utils.config import (
    BOOT_N,
    METRICS,
    MWU_ALTERNATIVE,
    REP_W,
    RNG_SEED,
)

mannwhitneyu: Callable[..., object] | None = None

try:
    from scipy.stats import mannwhitneyu as _mannwhitneyu

    mannwhitneyu = _mannwhitneyu
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


def separation_score(
    dist_df: pd.DataFrame,
    weights: Dict[str, float] = REP_W,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    gi = dist_df["group"] == "interacting"
    gn = dist_df["group"] == "non-interacting"

    for m in METRICS:
        med_i = float(np.nanmedian(dist_df.loc[gi, m])) if bool(gi.any()) else 0.0
        med_n = float(np.nanmedian(dist_df.loc[gn, m])) if bool(gn.any()) else 0.0
        out[f"diff_med_{m}"] = med_i - med_n

    out["sep_score"] = sum(weights[k] * out[f"diff_med_{k}"] for k in weights)
    return out


def sample_non_set(non_list: List[int], target_len: int) -> List[int]:
    if len(non_list) <= target_len:
        return [int(x) for x in non_list]

    rng = np.random.default_rng(RNG_SEED)
    idx = rng.choice(len(non_list), size=target_len, replace=False)
    return [int(non_list[int(i)]) for i in idx]


def mwu_pvalue(
    a: np.ndarray,
    b: np.ndarray,
    alternative: str = MWU_ALTERNATIVE,
) -> float:
    if len(a) == 0 or len(b) == 0:
        return 1.0

    if _HAVE_SCIPY and mannwhitneyu is not None:
        try:
            result = mannwhitneyu(a, b, alternative=alternative)
            pvalue = getattr(result, "pvalue", 1.0)
            return float(cast(float, pvalue))
        except Exception:
            pass

    rng = np.random.default_rng(0)
    allv = np.concatenate([a, b])
    obs = float(np.median(a) - np.median(b))
    cnt = 0
    iters = 2000

    for _ in range(iters):
        rng.shuffle(allv)
        aa = allv[: len(a)]
        bb = allv[len(a) :]
        if float(np.median(aa) - np.median(bb)) >= obs:
            cnt += 1

    return (cnt + 1) / (iters + 1)


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    a_arr = np.asarray(a)
    b_arr = np.asarray(b)
    na, nb = len(a_arr), len(b_arr)

    if na == 0 or nb == 0:
        return 0.0

    gt = sum(1 for x in a_arr for y in b_arr if x > y)
    lt = sum(1 for x in a_arr for y in b_arr if x < y)
    return (gt - lt) / float(na * nb)


def benjamini_hochberg(pvals: List[float]) -> List[float]:
    m = len(pvals)
    idx = np.argsort(pvals)
    ranked = np.array(pvals, dtype=float)[idx]
    q = np.empty(m, dtype=float)
    prev = 1.0

    for i in range(m - 1, -1, -1):
        rank = i + 1
        val = (m / rank) * ranked[i]
        prev = min(prev, float(val))
        q[i] = prev

    out = np.empty(m, dtype=float)
    out[idx] = q
    return out.tolist()


def summarize_metric_stats(
    dist_df: pd.DataFrame,
    metrics: Tuple[str, ...] = METRICS,
    alternative: str = MWU_ALTERNATIVE,
) -> Dict[str, Dict[str, float]]:
    res: Dict[str, Dict[str, float]] = {}
    gi = dist_df["group"] == "interacting"
    gn = dist_df["group"] == "non-interacting"

    for m in metrics:
        a = dist_df.loc[gi, m].dropna().to_numpy()
        b = dist_df.loc[gn, m].dropna().to_numpy()

        dm = float(np.nanmean(a) - np.nanmean(b)) if len(a) and len(b) else 0.0
        dmed = float(np.nanmedian(a) - np.nanmedian(b)) if len(a) and len(b) else 0.0
        p = mwu_pvalue(a, b, alternative=alternative)
        d = cliffs_delta(a, b)

        res[m] = {
            "d_mean": dm,
            "d_median": dmed,
            "p_mwu": p,
            "cliffs_delta": d,
        }

    return res


def bootstrap_mean_diff(
    a: np.ndarray,
    b: np.ndarray,
    n_boot: int = BOOT_N,
    seed: int = RNG_SEED,
) -> Tuple[float, float]:
    if len(a) == 0 or len(b) == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    ia = rng.integers(0, len(a), size=(n_boot, len(a)))
    ib = rng.integers(0, len(b), size=(n_boot, len(b)))
    diffs = np.nanmean(a[ia], axis=1) - np.nanmean(b[ib], axis=1)
    lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
    return float(lo), float(hi)
