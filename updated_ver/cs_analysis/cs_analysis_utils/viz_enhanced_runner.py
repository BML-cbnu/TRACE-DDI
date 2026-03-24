"""
viz_enhanced_runner.py
"""

from __future__ import annotations

import os
from typing import Protocol

import pandas as pd

from cs_analysis_utils.io_utils import ensure_dir, safe_fname, short_label
from cs_analysis_utils.viz_graphs_enhanced import draw_merged_graph_enhanced
from cs_analysis_utils.viz_utils import pick_representative_B_for_graph


class ResolverProtocol(Protocol):
    def drug_label(self, x: int) -> str: ...
    def pathway_label(self, node_info: object) -> str: ...


def visualize_primary_pairs_enhanced(
    primary_df: pd.DataFrame,
    ddi: pd.DataFrame,
    subg_base: str,
    resolver: ResolverProtocol,
    out_viz_stage_dir: str,
    stage_tag: str,
    max_labels: int = 25,
) -> None:
    if primary_df.empty:
        return

    enhanced_root = os.path.join(out_viz_stage_dir, "enhanced_graphs")
    ensure_dir(enhanced_root)

    for _, row in primary_df.iterrows():
        A = int(row["anchor"])
        y = int(row["interaction"])
        X = str(row["pathway"])

        anchor_dir = os.path.join(enhanced_root, f"A{A}")
        ensure_dir(anchor_dir)

        niceX = safe_fname(short_label(X, 60))
        b = pick_representative_B_for_graph(A, y, X, ddi, subg_base)
        if b is None:
            continue

        save_path = os.path.join(
            anchor_dir,
            f"EnhancedGraph_A{A}_Y{y}_B{int(b)}_X_{niceX}.png",
        )

        draw_merged_graph_enhanced(
            drug_a=A,
            drug_b=int(b),
            base_dir=subg_base,
            save_path=save_path,
            resolver=resolver,
            highlight_pathway=X,
            interaction_y=y,
            max_labels=max_labels,
        )
