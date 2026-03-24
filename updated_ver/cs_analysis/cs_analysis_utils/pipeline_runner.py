"""
pipeline_runner.py
"""

from __future__ import annotations

import os
from argparse import Namespace
from typing import Any

import pandas as pd

from cs_analysis_utils.config import (
    CASE_EXTENDED,
    CASE_FILTERED,
    CASE_SIMPLE,
    CLIFFS_DELTA_THR,
    FDR_THR,
    KEY_METRICS,
    METRICS_EXT,
    MIN_GROUP_SIZE,
    MIN_SHARE_RATIO,
    MWU_ALTERNATIVE,
    SAVE_TOGGLE,
    TOP_N_MAIN,
    TOPK_PER_ANCHOR,
)
from cs_analysis_utils.ddi_utils import calculate_portion
from cs_analysis_utils.io_utils import OutPaths, assert_ddi_types, read_ddi
from cs_analysis_utils.name_resolver import NameResolver
from cs_analysis_utils.ranking import (
    evaluate_anchor_label_fast,
    evaluate_anchor_label_fast_ext,
    identify_signal_pairs,
    identify_signal_pairs_ext,
    run_simple_wilcoxon,
)
from cs_analysis_utils.viz_enhanced_runner import visualize_primary_pairs_enhanced
from cs_analysis_utils.viz_utils import (
    visualize_primary_pairs,
    visualize_simple_wilcoxon,
)


def run_rank(
    ddi: pd.DataFrame,
    anchors_df: pd.DataFrame,
    subg_base: str,
) -> pd.DataFrame:
    from tqdm import tqdm

    rows: list[pd.DataFrame] = []
    anchor_pairs = anchors_df.loc[:, ["drug", "interaction"]]

    for drug_raw, interaction_raw in tqdm(
        anchor_pairs.itertuples(index=False, name=None),
        total=len(anchor_pairs),
        desc="rank",
    ):
        anchor = int(drug_raw)
        interaction = int(interaction_raw)

        part = evaluate_anchor_label_fast(
            ddi=ddi,
            anchor_a=anchor,
            label_y=interaction,
            base_dir=subg_base,
            min_group=MIN_GROUP_SIZE,
            top_k_pathways=TOPK_PER_ANCHOR,
        )
        if not part.empty:
            rows.append(part)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def run_rank_ext(
    ddi: pd.DataFrame,
    anchors_df: pd.DataFrame,
    subg_base: str,
) -> pd.DataFrame:
    from tqdm import tqdm

    rows: list[pd.DataFrame] = []
    anchor_pairs = anchors_df.loc[:, ["drug", "interaction"]]

    for drug_raw, interaction_raw in tqdm(
        anchor_pairs.itertuples(index=False, name=None),
        total=len(anchor_pairs),
        desc="rank_ext",
    ):
        anchor = int(drug_raw)
        interaction = int(interaction_raw)

        part = evaluate_anchor_label_fast_ext(
            ddi=ddi,
            anchor_a=anchor,
            label_y=interaction,
            base_dir=subg_base,
            min_group=MIN_GROUP_SIZE,
            top_k_pathways=TOPK_PER_ANCHOR,
        )
        if not part.empty:
            rows.append(part)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def _save_rank_csv(rank_df: pd.DataFrame, out_csv: str) -> None:
    if rank_df.empty:
        return

    rank_to_save = rank_df.copy()
    rank_to_save["anchor"] = rank_to_save["anchor"].astype(int)
    rank_to_save["interaction"] = rank_to_save["interaction"].astype(int)
    rank_to_save = rank_to_save.sort_values("final_score", ascending=False)
    rank_to_save.to_csv(out_csv, index=False)


def _save_stage_csv(
    df: pd.DataFrame,
    out_paths: OutPaths,
    case_name: str,
    default_stage: str,
    filename: str,
) -> None:
    if df.empty:
        return

    stage = str(df.iloc[0].get("selection_stage", default_stage))
    stats_dir = out_paths.stats_dir(case_name, stage=stage)
    df.to_csv(os.path.join(stats_dir, filename), index=False)


def _run_primary_visualization(
    primary_df: pd.DataFrame,
    ddi: pd.DataFrame,
    subg_base: str,
    resolver: NameResolver,
    out_paths: OutPaths,
    case_name: str,
    stage_tag_prefix: str = "",
    save_enhanced_graphs: bool = False,
    enhanced_max_labels: int = 25,
) -> None:
    if primary_df.empty:
        return

    stage = str(primary_df.iloc[0].get("selection_stage", "strict"))
    viz_stage_dir = out_paths.viz_dir(case_name, stage=stage)

    stage_tag = f"{stage_tag_prefix}{stage}" if stage_tag_prefix else stage

    visualize_primary_pairs(
        primary_df=primary_df,
        ddi=ddi,
        subg_base=subg_base,
        resolver=resolver,
        out_viz_stage_dir=viz_stage_dir,
        stage_tag=stage_tag,
    )

    if save_enhanced_graphs:
        visualize_primary_pairs_enhanced(
            primary_df=primary_df,
            ddi=ddi,
            subg_base=subg_base,
            resolver=resolver,
            out_viz_stage_dir=viz_stage_dir,
            stage_tag=stage_tag,
            max_labels=enhanced_max_labels,
        )


def run_pipeline(args: Namespace | Any) -> int:
    ddi = read_ddi(args.ddi)
    assert_ddi_types(ddi)

    resolver = NameResolver(
        drkg_nodes_path=args.drkg_nodes,
        hetionet_nodes_path=args.hetionet_nodes,
    )
    out_paths = OutPaths(args.save_root)
    anchor_df = calculate_portion(ddi)

    save_enhanced_graphs = bool(getattr(args, "save_enhanced_graphs", False))
    enhanced_max_labels = int(getattr(args, "enhanced_max_labels", 25))

    # --------------------------------------------------
    # Case 1: filtered_wilcoxon
    # --------------------------------------------------
    rank_all = run_rank(ddi=ddi, anchors_df=anchor_df, subg_base=args.subg_base)

    if SAVE_TOGGLE and not rank_all.empty:
        _save_rank_csv(
            rank_df=rank_all,
            out_csv=os.path.join(out_paths.case_root(CASE_FILTERED), "rank_all.csv"),
        )

    if rank_all.empty:
        primary_pairs = pd.DataFrame()
        screened_pairs = pd.DataFrame()
    else:
        primary_pairs, screened_pairs = identify_signal_pairs(
            rank_all=rank_all,
            ddi=ddi,
            base_dir=args.subg_base,
            delta_thr=CLIFFS_DELTA_THR,
            fdr_thr=FDR_THR,
            min_each_group=MIN_GROUP_SIZE,
            min_coverage=MIN_SHARE_RATIO,
            require_two_metrics=KEY_METRICS,
            top_n_primary=TOP_N_MAIN,
            enable_fallback_when_empty=True,
            use_or_on_keys=False,
            min_n_sig_metrics=2,
            mwu_alternative=MWU_ALTERNATIVE,
        )

    if SAVE_TOGGLE:
        _save_stage_csv(
            df=screened_pairs,
            out_paths=out_paths,
            case_name=CASE_FILTERED,
            default_stage="strict",
            filename="signal_pairs_FDR.csv",
        )
        _save_stage_csv(
            df=primary_pairs,
            out_paths=out_paths,
            case_name=CASE_FILTERED,
            default_stage="strict",
            filename="primary_signals.csv",
        )

    _run_primary_visualization(
        primary_df=primary_pairs,
        ddi=ddi,
        subg_base=args.subg_base,
        resolver=resolver,
        out_paths=out_paths,
        case_name=CASE_FILTERED,
        stage_tag_prefix="",
        save_enhanced_graphs=save_enhanced_graphs,
        enhanced_max_labels=enhanced_max_labels,
    )

    # --------------------------------------------------
    # Case 2: simple_wilcoxon
    # --------------------------------------------------
    if not rank_all.empty:
        wdf = run_simple_wilcoxon(
            rank_all=rank_all,
            ddi=ddi,
            base_dir=args.subg_base,
            min_each_group=MIN_GROUP_SIZE,
            mwu_alternative=MWU_ALTERNATIVE,
        )

        if SAVE_TOGGLE and not wdf.empty:
            simple_stats_dir = out_paths.stats_dir(CASE_SIMPLE)
            wdf.to_csv(
                os.path.join(simple_stats_dir, "wilcoxon_results.csv"),
                index=False,
            )

            figs_root = out_paths.figs_dir(CASE_SIMPLE)
            visualize_simple_wilcoxon(
                wdf=wdf,
                ddi=ddi,
                subg_base=args.subg_base,
                resolver=resolver,
                figs_root=figs_root,
            )

    # --------------------------------------------------
    # Case 3: extended
    # --------------------------------------------------
    rank_all_ext = run_rank_ext(
        ddi=ddi,
        anchors_df=anchor_df,
        subg_base=args.subg_base,
    )

    if SAVE_TOGGLE and not rank_all_ext.empty:
        _save_rank_csv(
            rank_df=rank_all_ext,
            out_csv=os.path.join(
                out_paths.case_root(CASE_EXTENDED),
                "rank_all_ext.csv",
            ),
        )

    if rank_all_ext.empty:
        primary_pairs_ext = pd.DataFrame()
        screened_pairs_ext = pd.DataFrame()
    else:
        primary_pairs_ext, screened_pairs_ext = identify_signal_pairs_ext(
            rank_all=rank_all_ext,
            ddi=ddi,
            base_dir=args.subg_base,
            metrics=METRICS_EXT,
            delta_thr=CLIFFS_DELTA_THR,
            fdr_thr=FDR_THR,
            min_each_group=MIN_GROUP_SIZE,
            min_coverage=MIN_SHARE_RATIO,
            require_two_metrics=KEY_METRICS,
            top_n_primary=TOP_N_MAIN,
            enable_fallback_when_empty=True,
            use_or_on_keys=False,
            min_n_sig_metrics=2,
            mwu_alternative=MWU_ALTERNATIVE,
        )

    if SAVE_TOGGLE:
        _save_stage_csv(
            df=screened_pairs_ext,
            out_paths=out_paths,
            case_name=CASE_EXTENDED,
            default_stage="strict",
            filename="signal_candidates_ext.csv",
        )
        _save_stage_csv(
            df=primary_pairs_ext,
            out_paths=out_paths,
            case_name=CASE_EXTENDED,
            default_stage="strict",
            filename="primary_signals_ext.csv",
        )

    _run_primary_visualization(
        primary_df=primary_pairs_ext,
        ddi=ddi,
        subg_base=args.subg_base,
        resolver=resolver,
        out_paths=out_paths,
        case_name=CASE_EXTENDED,
        stage_tag_prefix="extended-",
        save_enhanced_graphs=save_enhanced_graphs,
        enhanced_max_labels=enhanced_max_labels,
    )

    return 0
