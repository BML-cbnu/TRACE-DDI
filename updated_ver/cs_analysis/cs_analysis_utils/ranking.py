from __future__ import annotations

from typing import Iterable, Sequence, cast

import numpy as np
import pandas as pd

from cs_analysis_utils.config import (
    CLIFFS_DELTA_THR,
    FDR_THR,
    KEY_METRICS,
    METRICS,
    METRICS_EXT,
    MIN_GROUP_SIZE,
    MIN_SHARE_RATIO,
    MWU_ALTERNATIVE,
    NON_RATIO,
    TOP_N_MAIN,
    TOPK_PER_ANCHOR,
)
from cs_analysis_utils.ddi_utils import (
    B_sets_for_label,
    drug_has_pathway,
    pathway_candidates_shared,
)
from cs_analysis_utils.extended_metrics import (
    compute_extra_metrics,
    separation_score_ext,
)
from cs_analysis_utils.graph_io import (
    centralities_for_all_nodes_cached,
    load_graph_files_cached,
    merged_graph_cached,
)
from cs_analysis_utils.stats_utils import (
    benjamini_hochberg,
    sample_non_set,
    separation_score,
    summarize_metric_stats,
)

MetricSeq = Sequence[str]
PerRowEntry = tuple[tuple[int, int, str], dict[str, dict[str, float]], int, int, float]


def _metric_columns(metrics: MetricSeq) -> list[str]:
    cols: list[str] = []
    for m in metrics:
        cols.append(f"{m}_q")
        cols.append(f"{m}_d_mean")
        cols.append(f"{m}_delta")
    return cols


def _empty_signal_frames(metrics: MetricSeq) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = [
        "anchor",
        "interaction",
        "pathway",
        "n_int",
        "n_non",
        "coverage_ratio",
        "is_primary",
        "selection_stage",
        "notes",
        *_metric_columns(metrics),
        "n_sig_metrics",
        "sum_d_mean",
        "min_q",
    ]
    empty_df = pd.DataFrame(columns=pd.Index(cols, dtype="object"))
    return empty_df.copy(), empty_df.copy()


def _get_cov_ratio(
    rank_all: pd.DataFrame,
    anchor: int,
    interaction: int,
    pathway: str,
) -> float:
    series = rank_all.loc[
        (rank_all["anchor"] == anchor)
        & (rank_all["interaction"] == interaction)
        & (rank_all["pathway"] == pathway),
        "coverage_ratio",
    ]
    if len(series) == 0:
        return 0.0
    return float(series.iloc[0])


def _build_fallback_df(
    rank_all: pd.DataFrame,
    metrics: MetricSeq,
    top_n_primary: int,
) -> pd.DataFrame:
    fallback_base = (
        rank_all.sort_values("final_score", ascending=False)
        .loc[:, ["anchor", "interaction", "pathway"]]
        .drop_duplicates()
        .head(top_n_primary)
        .copy()
    )
    fallback = cast(pd.DataFrame, fallback_base)

    for c in [
        "n_int",
        "n_non",
        "coverage_ratio",
        "is_primary",
        "selection_stage",
        "notes",
        "n_sig_metrics",
        "sum_d_mean",
        "min_q",
    ]:
        fallback[c] = np.nan

    for m in metrics:
        fallback[f"{m}_q"] = np.nan
        fallback[f"{m}_d_mean"] = np.nan
        fallback[f"{m}_delta"] = np.nan

    fallback["is_primary"] = False
    fallback["selection_stage"] = "fallback"
    fallback["notes"] = "[fallback] ranked by final_score; not statistically selected"
    return fallback


def _shared_partner_sets(
    ddi: pd.DataFrame,
    anchor: int,
    interaction: int,
    pathway: str,
    base_dir: str,
) -> tuple[list[int], list[int]]:
    B_int, B_non = B_sets_for_label(ddi, anchor, interaction)
    B_int_shared = [
        int(b) for b in B_int if drug_has_pathway(int(b), base_dir, pathway)
    ]
    B_non_shared = [
        int(b) for b in B_non if drug_has_pathway(int(b), base_dir, pathway)
    ]
    return B_int_shared, B_non_shared


def measure_X_over_B_fast(
    anchor_a: int,
    pathway: str,
    B_list: Iterable[int],
    base_dir: str,
    require_shared: bool = True,
) -> pd.DataFrame:
    Ae, _ = load_graph_files_cached(int(anchor_a), base_dir)
    if Ae is None:
        return pd.DataFrame(columns=pd.Index(["drug_b", *METRICS], dtype="object"))

    out: list[dict[str, object]] = []

    for b_raw in B_list:
        b = int(b_raw)
        if require_shared and not drug_has_pathway(b, base_dir, pathway):
            continue

        Be, _ = load_graph_files_cached(b, base_dir)
        if Be is None:
            continue

        G = merged_graph_cached(Ae, Be, int(anchor_a), b)
        c = centralities_for_all_nodes_cached(G, int(anchor_a), b)
        if pathway not in c["degree"]:
            continue

        out.append(
            {
                "drug_b": b,
                "degree": float(c["degree"].get(pathway, 0.0)),
                "betweenness": float(c["betweenness"].get(pathway, 0.0)),
                "closeness": float(c["closeness"].get(pathway, 0.0)),
                "eigenvector": float(c["eigenvector"].get(pathway, 0.0)),
            }
        )

    return pd.DataFrame(out)


def measure_X_over_B_fast_ext(
    anchor_a: int,
    pathway: str,
    B_list: Iterable[int],
    base_dir: str,
    require_shared: bool = True,
) -> pd.DataFrame:
    Ae, _ = load_graph_files_cached(int(anchor_a), base_dir)
    if Ae is None:
        return pd.DataFrame(columns=pd.Index(["drug_b", *METRICS_EXT], dtype="object"))

    out: list[dict[str, object]] = []

    for b_raw in B_list:
        b = int(b_raw)
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

        out.append(
            {
                "drug_b": b,
                "degree": float(base["degree"].get(pathway, 0.0)),
                "betweenness": float(base["betweenness"].get(pathway, 0.0)),
                "closeness": float(base["closeness"].get(pathway, 0.0)),
                "eigenvector": float(base["eigenvector"].get(pathway, 0.0)),
                "clustering_inv": float(extra["clustering_inv"].get(pathway, 0.0)),
                "eccentricity_inv": float(extra["eccentricity_inv"].get(pathway, 0.0)),
                "radiality": float(extra["radiality"].get(pathway, 0.0)),
            }
        )

    return pd.DataFrame(out)


def evaluate_anchor_label_fast(
    ddi: pd.DataFrame,
    anchor_a: int,
    label_y: int,
    base_dir: str,
    min_group: int = MIN_GROUP_SIZE,
    top_k_pathways: int = TOPK_PER_ANCHOR,
    non_ratio: float = NON_RATIO,
    min_share_ratio: float = MIN_SHARE_RATIO,
) -> pd.DataFrame:
    B_int, B_non = B_sets_for_label(ddi, int(anchor_a), int(label_y))
    if len(B_int) < min_group or len(B_non) < min_group:
        return pd.DataFrame()

    candX, coverage = pathway_candidates_shared(
        int(anchor_a),
        int(label_y),
        base_dir,
        ddi,
        min_share_ratio=min_share_ratio,
    )
    if not candX:
        return pd.DataFrame()

    non_target_len = int(min(len(B_non), max(min_group, int(non_ratio * len(B_int)))))
    B_non_s = sample_non_set(B_non, non_target_len)

    rows: list[dict[str, object]] = []

    for X in candX:
        df_i = measure_X_over_B_fast(
            int(anchor_a), X, B_int, base_dir, require_shared=True
        )
        df_n = measure_X_over_B_fast(
            int(anchor_a), X, B_non_s, base_dir, require_shared=True
        )
        if df_i.empty or df_n.empty:
            continue

        df_i = df_i.copy()
        df_n = df_n.copy()
        df_i["group"] = "interacting"
        df_n["group"] = "non-interacting"

        dist = pd.concat([df_i, df_n], ignore_index=True)
        sep = separation_score(dist)
        med_eig_int = (
            float(np.nanmedian(df_i["eigenvector"])) if "eigenvector" in df_i else 0.0
        )

        cov_count, cov_ratio = coverage.get(X, (0, 0.0))
        final = (
            sep["sep_score"]
            * (med_eig_int if med_eig_int > 0 else 1.0)
            * (1.0 + cov_ratio)
        )

        row: dict[str, object] = {
            "anchor": int(anchor_a),
            "interaction": int(label_y),
            "pathway": X,
            "n_int": int(len(df_i)),
            "n_non": int(len(df_n)),
            "coverage_count": int(cov_count),
            "coverage_ratio": float(cov_ratio),
            "median_eig_int": float(med_eig_int),
            "final_score": float(final),
        }
        row.update(sep)
        rows.append(row)

    res = pd.DataFrame(rows)
    if res.empty:
        return res

    return cast(
        pd.DataFrame,
        res.sort_values("final_score", ascending=False)
        .head(top_k_pathways)
        .reset_index(drop=True),
    )


def evaluate_anchor_label_fast_ext(
    ddi: pd.DataFrame,
    anchor_a: int,
    label_y: int,
    base_dir: str,
    min_group: int = MIN_GROUP_SIZE,
    top_k_pathways: int = TOPK_PER_ANCHOR,
    non_ratio: float = NON_RATIO,
    min_share_ratio: float = MIN_SHARE_RATIO,
) -> pd.DataFrame:
    B_int, B_non = B_sets_for_label(ddi, int(anchor_a), int(label_y))
    if len(B_int) < min_group or len(B_non) < min_group:
        return pd.DataFrame()

    candX, coverage = pathway_candidates_shared(
        int(anchor_a),
        int(label_y),
        base_dir,
        ddi,
        min_share_ratio=min_share_ratio,
    )
    if not candX:
        return pd.DataFrame()

    non_target_len = int(min(len(B_non), max(min_group, int(non_ratio * len(B_int)))))
    B_non_s = sample_non_set(B_non, non_target_len)

    rows: list[dict[str, object]] = []

    for X in candX:
        df_i = measure_X_over_B_fast_ext(
            int(anchor_a), X, B_int, base_dir, require_shared=True
        )
        df_n = measure_X_over_B_fast_ext(
            int(anchor_a), X, B_non_s, base_dir, require_shared=True
        )
        if df_i.empty or df_n.empty:
            continue

        df_i = df_i.copy()
        df_n = df_n.copy()
        df_i["group"] = "interacting"
        df_n["group"] = "non-interacting"

        dist = pd.concat([df_i, df_n], ignore_index=True)
        sep = separation_score_ext(dist)
        med_eig_int = (
            float(np.nanmedian(df_i["eigenvector"])) if "eigenvector" in df_i else 0.0
        )

        cov_count, cov_ratio = coverage.get(X, (0, 0.0))
        final = (
            sep["sep_score"]
            * (med_eig_int if med_eig_int > 0 else 1.0)
            * (1.0 + cov_ratio)
        )

        row: dict[str, object] = {
            "anchor": int(anchor_a),
            "interaction": int(label_y),
            "pathway": X,
            "n_int": int(len(df_i)),
            "n_non": int(len(df_n)),
            "coverage_count": int(cov_count),
            "coverage_ratio": float(cov_ratio),
            "median_eig_int": float(med_eig_int),
            "final_score": float(final),
        }
        row.update(sep)
        rows.append(row)

    res = pd.DataFrame(rows)
    if res.empty:
        return res

    return cast(
        pd.DataFrame,
        res.sort_values("final_score", ascending=False)
        .head(top_k_pathways)
        .reset_index(drop=True),
    )


def _collect_per_row_metrics(
    rank_all: pd.DataFrame,
    ddi: pd.DataFrame,
    base_dir: str,
    metrics: MetricSeq,
    use_extended_measure: bool,
    min_each_group: int,
    min_coverage: float,
    mwu_alternative: str,
) -> tuple[list[float], list[PerRowEntry]]:
    tmp_pvals: list[float] = []
    per_row_metrics: list[PerRowEntry] = []

    combos = rank_all.loc[:, ["anchor", "interaction", "pathway"]]

    for A, y, X in combos.itertuples(index=False, name=None):
        A_i = int(A)
        y_i = int(y)
        X_s = str(X)

        B_int_shared, B_non_shared = _shared_partner_sets(ddi, A_i, y_i, X_s, base_dir)
        cov_ratio = _get_cov_ratio(rank_all, A_i, y_i, X_s)

        cov_ok = cov_ratio >= min_coverage
        size_ok = (
            len(B_int_shared) >= min_each_group and len(B_non_shared) >= min_each_group
        )
        if not (cov_ok and size_ok):
            continue

        if use_extended_measure:
            df_i = measure_X_over_B_fast_ext(
                A_i, X_s, B_int_shared, base_dir, require_shared=True
            )
            df_n = measure_X_over_B_fast_ext(
                A_i, X_s, B_non_shared, base_dir, require_shared=True
            )
        else:
            df_i = measure_X_over_B_fast(
                A_i, X_s, B_int_shared, base_dir, require_shared=True
            )
            df_n = measure_X_over_B_fast(
                A_i, X_s, B_non_shared, base_dir, require_shared=True
            )

        if df_i.empty or df_n.empty:
            continue

        df_i = df_i.copy()
        df_n = df_n.copy()
        df_i["group"] = "interacting"
        df_n["group"] = "non-interacting"
        dist_df = pd.concat([df_i, df_n], ignore_index=True)

        stats = cast(
            dict[str, dict[str, float]],
            summarize_metric_stats(
                dist_df,
                metrics=tuple(metrics),
                alternative=mwu_alternative,
            ),
        )

        per_row_metrics.append(
            ((A_i, y_i, X_s), stats, len(df_i), len(df_n), cov_ratio)
        )
        for m in metrics:
            tmp_pvals.append(float(stats[m]["p_mwu"]))

    return tmp_pvals, per_row_metrics


def _strict_select_rows(
    per_row_metrics: list[PerRowEntry],
    qvals: list[float],
    metrics: MetricSeq,
    require_two_metrics: Sequence[str],
    fdr_thr: float,
    delta_thr: float,
    use_or_on_keys: bool,
    min_n_sig_metrics: int,
) -> list[dict[str, object]]:
    sig_rows: list[dict[str, object]] = []
    qi = 0

    for (A, y, X), stats, n_i, n_n, cov in per_row_metrics:
        mstats: dict[str, dict[str, float]] = {}
        min_q_val = 1.0
        sum_dmean = 0.0

        for m in metrics:
            p = float(stats[m]["p_mwu"])
            q = float(qvals[qi])
            qi += 1
            dm = float(stats[m]["d_mean"])
            d = float(stats[m]["cliffs_delta"])

            mstats[m] = {"p": p, "q": q, "d_mean": dm, "cliffs_delta": d}
            min_q_val = min(min_q_val, q)
            sum_dmean += max(0.0, dm)

        sig_count = sum(
            (mstats[k]["q"] < fdr_thr)
            and (mstats[k]["cliffs_delta"] >= delta_thr)
            and (mstats[k]["d_mean"] > 0.0)
            for k in require_two_metrics
            if k in mstats
        )

        pass_keys = sig_count >= 1 if use_or_on_keys else sig_count >= min_n_sig_metrics
        if not pass_keys:
            continue

        row: dict[str, object] = {
            "anchor": int(A),
            "interaction": int(y),
            "pathway": X,
            "n_int": int(n_i),
            "n_non": int(n_n),
            "coverage_ratio": float(cov),
            "n_sig_metrics": int(sig_count),
            "sum_d_mean": float(sum_dmean),
            "min_q": float(min_q_val),
            "is_primary": False,
            "selection_stage": "strict",
            "notes": f"[strict] k-of-{len(require_two_metrics)}: {sig_count} met",
        }

        for m in metrics:
            row[f"{m}_q"] = float(mstats[m]["q"])
            row[f"{m}_d_mean"] = float(mstats[m]["d_mean"])
            row[f"{m}_delta"] = float(mstats[m]["cliffs_delta"])

        sig_rows.append(row)

    return sig_rows


def _relaxed_select_rows(
    per_row_metrics: list[PerRowEntry],
    metrics: MetricSeq,
    min_abs_dmean: dict[str, float],
    loose_p_thr: float,
    loose_delta: float,
) -> list[dict[str, object]]:
    loose_rows: list[dict[str, object]] = []

    for (A, y, X), stats, n_i, n_n, cov in per_row_metrics:
        passed_any = False
        mstats: dict[str, dict[str, float]] = {}
        sum_dmean = 0.0
        n_passed = 0

        for m in metrics:
            p = float(stats[m]["p_mwu"])
            dm = float(stats[m]["d_mean"])
            d = float(stats[m]["cliffs_delta"])

            cond = (
                (dm > min_abs_dmean.get(m, 0.0))
                and (d >= loose_delta)
                and (p < loose_p_thr)
            )
            if cond:
                n_passed += 1

            passed_any = passed_any or cond
            mstats[m] = {"q": p, "d_mean": dm, "cliffs_delta": d}
            sum_dmean += max(0.0, dm)

        if not (
            passed_any
            and (n_i >= MIN_GROUP_SIZE)
            and (n_n >= MIN_GROUP_SIZE)
            and (cov >= MIN_SHARE_RATIO)
        ):
            continue

        row: dict[str, object] = {
            "anchor": int(A),
            "interaction": int(y),
            "pathway": X,
            "n_int": int(n_i),
            "n_non": int(n_n),
            "coverage_ratio": float(cov),
            "n_sig_metrics": int(n_passed),
            "sum_d_mean": float(sum_dmean),
            "min_q": float(min(mstats[m]["q"] for m in metrics)),
            "is_primary": False,
            "selection_stage": "relaxed",
            "notes": "[relaxed] p<0.10 & d_mean>min & δ≥0.05",
        }

        for m in metrics:
            row[f"{m}_q"] = float(mstats[m]["q"])
            row[f"{m}_d_mean"] = float(mstats[m]["d_mean"])
            row[f"{m}_delta"] = float(mstats[m]["cliffs_delta"])

        loose_rows.append(row)

    return loose_rows


def identify_signal_pairs(
    rank_all: pd.DataFrame,
    ddi: pd.DataFrame,
    base_dir: str,
    delta_thr: float = CLIFFS_DELTA_THR,
    fdr_thr: float = FDR_THR,
    min_each_group: int = MIN_GROUP_SIZE,
    min_coverage: float = MIN_SHARE_RATIO,
    require_two_metrics: tuple[str, ...] = KEY_METRICS,
    top_n_primary: int = TOP_N_MAIN,
    enable_fallback_when_empty: bool = True,
    use_or_on_keys: bool = False,
    min_n_sig_metrics: int = 2,
    mwu_alternative: str = MWU_ALTERNATIVE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tmp_pvals, per_row_metrics = _collect_per_row_metrics(
        rank_all=rank_all,
        ddi=ddi,
        base_dir=base_dir,
        metrics=METRICS,
        use_extended_measure=False,
        min_each_group=min_each_group,
        min_coverage=min_coverage,
        mwu_alternative=mwu_alternative,
    )

    if not tmp_pvals:
        return _empty_signal_frames(METRICS)

    qvals = benjamini_hochberg(tmp_pvals)
    sig_rows = _strict_select_rows(
        per_row_metrics=per_row_metrics,
        qvals=qvals,
        metrics=METRICS,
        require_two_metrics=require_two_metrics,
        fdr_thr=fdr_thr,
        delta_thr=delta_thr,
        use_or_on_keys=use_or_on_keys,
        min_n_sig_metrics=min_n_sig_metrics,
    )

    if len(sig_rows) > 0:
        sig_df = cast(
            pd.DataFrame,
            pd.DataFrame(sig_rows).sort_values(
                ["n_sig_metrics", "sum_d_mean", "min_q"],
                ascending=[False, False, True],
            ),
        )
        primary_df = cast(pd.DataFrame, sig_df.head(top_n_primary).copy())
        primary_df.loc[:, "is_primary"] = True
        return primary_df, sig_df

    loose_rows = _relaxed_select_rows(
        per_row_metrics=per_row_metrics,
        metrics=METRICS,
        min_abs_dmean={
            "degree": 0.001,
            "betweenness": 1e-5,
            "closeness": 0.0005,
            "eigenvector": 1e-4,
        },
        loose_p_thr=0.10,
        loose_delta=0.05,
    )

    if len(loose_rows) > 0:
        sig_df = cast(
            pd.DataFrame,
            pd.DataFrame(loose_rows).sort_values(
                ["n_sig_metrics", "sum_d_mean", "min_q"],
                ascending=[False, False, True],
            ),
        )
        primary_df = cast(pd.DataFrame, sig_df.head(top_n_primary).copy())
        primary_df.loc[:, "is_primary"] = True
        return primary_df, sig_df

    if not enable_fallback_when_empty:
        return _empty_signal_frames(METRICS)

    fallback = _build_fallback_df(rank_all, METRICS, top_n_primary)
    return fallback.copy(), fallback.copy()


def identify_signal_pairs_ext(
    rank_all: pd.DataFrame,
    ddi: pd.DataFrame,
    base_dir: str,
    metrics: tuple[str, ...] = METRICS_EXT,
    delta_thr: float = CLIFFS_DELTA_THR,
    fdr_thr: float = FDR_THR,
    min_each_group: int = MIN_GROUP_SIZE,
    min_coverage: float = MIN_SHARE_RATIO,
    require_two_metrics: tuple[str, ...] = KEY_METRICS,
    top_n_primary: int = TOP_N_MAIN,
    enable_fallback_when_empty: bool = True,
    use_or_on_keys: bool = False,
    min_n_sig_metrics: int = 2,
    mwu_alternative: str = MWU_ALTERNATIVE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tmp_pvals, per_row_metrics = _collect_per_row_metrics(
        rank_all=rank_all,
        ddi=ddi,
        base_dir=base_dir,
        metrics=metrics,
        use_extended_measure=True,
        min_each_group=min_each_group,
        min_coverage=min_coverage,
        mwu_alternative=mwu_alternative,
    )

    if not tmp_pvals:
        return _empty_signal_frames(metrics)

    qvals = benjamini_hochberg(tmp_pvals)
    sig_rows = _strict_select_rows(
        per_row_metrics=per_row_metrics,
        qvals=qvals,
        metrics=metrics,
        require_two_metrics=require_two_metrics,
        fdr_thr=fdr_thr,
        delta_thr=delta_thr,
        use_or_on_keys=use_or_on_keys,
        min_n_sig_metrics=min_n_sig_metrics,
    )

    if len(sig_rows) > 0:
        sig_df = cast(
            pd.DataFrame,
            pd.DataFrame(sig_rows).sort_values(
                ["n_sig_metrics", "sum_d_mean", "min_q"],
                ascending=[False, False, True],
            ),
        )
        primary_df = cast(pd.DataFrame, sig_df.head(top_n_primary).copy())
        primary_df.loc[:, "is_primary"] = True
        return primary_df, sig_df

    loose_rows = _relaxed_select_rows(
        per_row_metrics=per_row_metrics,
        metrics=metrics,
        min_abs_dmean={
            "degree": 0.001,
            "betweenness": 1e-5,
            "closeness": 0.0005,
            "eigenvector": 1e-4,
            "clustering_inv": 1e-4,
            "eccentricity_inv": 1e-4,
            "radiality": 1e-4,
        },
        loose_p_thr=0.10,
        loose_delta=0.05,
    )

    if len(loose_rows) > 0:
        sig_df = cast(
            pd.DataFrame,
            pd.DataFrame(loose_rows).sort_values(
                ["n_sig_metrics", "sum_d_mean", "min_q"],
                ascending=[False, False, True],
            ),
        )
        primary_df = cast(pd.DataFrame, sig_df.head(top_n_primary).copy())
        primary_df.loc[:, "is_primary"] = True
        return primary_df, sig_df

    if not enable_fallback_when_empty:
        return _empty_signal_frames(metrics)

    fallback = _build_fallback_df(rank_all, metrics, top_n_primary)
    return fallback.copy(), fallback.copy()


def run_simple_wilcoxon(
    rank_all: pd.DataFrame,
    ddi: pd.DataFrame,
    base_dir: str,
    min_each_group: int = MIN_GROUP_SIZE,
    mwu_alternative: str = MWU_ALTERNATIVE,
) -> pd.DataFrame:
    if rank_all.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    all_pvals: list[float] = []
    stash: list[
        tuple[tuple[int, int, str], dict[str, dict[str, float]], int, int, float]
    ] = []

    combos = rank_all.loc[:, ["anchor", "interaction", "pathway"]].drop_duplicates()

    for A, y, X in combos.itertuples(index=False, name=None):
        anchor = int(A)
        interaction = int(y)
        pathway = str(X)

        B_int_shared, B_non_shared = _shared_partner_sets(
            ddi, anchor, interaction, pathway, base_dir
        )
        if len(B_int_shared) < min_each_group or len(B_non_shared) < min_each_group:
            continue

        df_i = measure_X_over_B_fast(
            anchor, pathway, B_int_shared, base_dir, require_shared=True
        )
        df_n = measure_X_over_B_fast(
            anchor, pathway, B_non_shared, base_dir, require_shared=True
        )
        if df_i.empty or df_n.empty:
            continue

        df_i = df_i.copy()
        df_n = df_n.copy()
        df_i["group"] = "interacting"
        df_n["group"] = "non-interacting"
        dist_df = pd.concat([df_i, df_n], ignore_index=True)

        stats = cast(
            dict[str, dict[str, float]],
            summarize_metric_stats(
                dist_df,
                metrics=METRICS,
                alternative=mwu_alternative,
            ),
        )

        cov_ratio = _get_cov_ratio(rank_all, anchor, interaction, pathway)

        stash.append(
            (
                (anchor, interaction, pathway),
                stats,
                int((dist_df["group"] == "interacting").sum()),
                int((dist_df["group"] == "non-interacting").sum()),
                cov_ratio,
            )
        )

        for m in METRICS:
            all_pvals.append(float(stats[m]["p_mwu"]))

    if not stash:
        return pd.DataFrame()

    qvals = benjamini_hochberg(all_pvals)
    qi = 0

    for (anchor, interaction, pathway), stats, n_i, n_n, cov_ratio in stash:
        row: dict[str, object] = {
            "anchor": anchor,
            "interaction": interaction,
            "pathway": pathway,
            "n_int": n_i,
            "n_non": n_n,
            "coverage_ratio": cov_ratio,
        }

        min_q = 1.0
        n_p_lt_005 = 0

        for m in METRICS:
            p = float(stats[m]["p_mwu"])
            q = float(qvals[qi])
            qi += 1

            row[f"{m}_p"] = p
            row[f"{m}_q"] = q
            row[f"{m}_d_mean"] = float(stats[m]["d_mean"])
            row[f"{m}_d_median"] = float(stats[m]["d_median"])

            min_q = min(min_q, q)
            n_p_lt_005 += int(p < 0.05)

        row["min_q"] = float(min_q)
        row["n_p_lt_0_05"] = int(n_p_lt_005)
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    return cast(
        pd.DataFrame,
        out.sort_values(["min_q", "n_p_lt_0_05"], ascending=[True, False]).reset_index(
            drop=True
        ),
    )
