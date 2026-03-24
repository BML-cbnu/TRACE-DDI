from __future__ import annotations

import os
from typing import Dict, Iterable, Optional, Protocol, Tuple, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

from cs_analysis_utils.config import (
    BOOT_N,
    EIG_MAX_ITER,
    EIG_TOL,
    METRICS,
    RNG_SEED,
    SAVE_TOGGLE,
)
from cs_analysis_utils.graph_io import (
    centralities_for_all_nodes_cached,
    load_graph_files_cached,
    merged_graph_cached,
)
from cs_analysis_utils.io_utils import ensure_dir, safe_fname, short_label
from cs_analysis_utils.ranking import measure_X_over_B_fast
from cs_analysis_utils.stats_utils import bootstrap_mean_diff

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["axes.grid"] = False
plt.rcParams["font.size"] = 10


class ResolverProtocol(Protocol):
    def drug_label(self, x: int) -> str: ...
    def pathway_label(self, node_info: object) -> str: ...


def _to_optional_int_scalar(x: object) -> Optional[int]:
    """Convert a scalar-like object to int if possible, otherwise None."""
    if x is None:
        return None

    # pandas missing
    try:
        if bool(pd.isna(x)):
            return None
    except Exception:
        pass

    # direct numeric path
    if isinstance(x, (int, np.integer)):
        return int(x)

    if isinstance(x, (float, np.floating)):
        if np.isnan(float(x)):
            return None
        return int(x)

    # conservative string parse
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return None

    try:
        return int(s)
    except Exception:
        try:
            return int(float(s))
        except Exception:
            return None


def _row_to_int(row: pd.Series, key: str) -> int:
    value = _to_optional_int_scalar(row[key])
    if value is None:
        raise ValueError(f"Cannot convert row[{key!r}] to int")
    return value


def _row_to_str(row: pd.Series, key: str) -> str:
    return str(row[key])


def _subset_group_df(dist_df: pd.DataFrame, group_name: str) -> pd.DataFrame:
    sub = dist_df.loc[dist_df["group"] == group_name, :].copy()
    return cast(pd.DataFrame, sub)


def node_name_to_nodeid_pref_node_id(
    nodes: Optional[pd.DataFrame],
) -> Dict[str, int]:
    if nodes is None or nodes.empty:
        return {}

    col = (
        "node_id"
        if "node_id" in nodes.columns
        else ("ori_node_id" if "ori_node_id" in nodes.columns else None)
    )
    if col is None:
        return {}

    out: Dict[str, int] = {}
    for _, row in nodes.iterrows():
        row_s = cast(pd.Series, row)
        node_id = _to_optional_int_scalar(row_s[col])
        if node_id is None:
            continue
        out[str(row_s["node_name"])] = node_id

    return out


def build_node_color_map(
    G: nx.Graph,
    An: Optional[pd.DataFrame],
    Bn: Optional[pd.DataFrame],
    Ae: Optional[pd.DataFrame],
    Be: Optional[pd.DataFrame],
    drug_a: int,
    drug_b: int,
    highlight: Optional[Iterable[str]] = None,
) -> Dict[str, str]:
    _ = Ae, Be

    highlight_set = set(highlight or [])
    mapA = node_name_to_nodeid_pref_node_id(An)
    mapB = node_name_to_nodeid_pref_node_id(Bn)

    colors: Dict[str, str] = {}
    for n in G.nodes():
        ns = str(n)

        if ns in highlight_set:
            colors[ns] = "#FFD54F"
            continue

        if (ns in mapA and mapA[ns] == drug_a) or (ns in mapB and mapB[ns] == drug_b):
            colors[ns] = "#EF5350"
            continue

        inA = ns in mapA
        inB = ns in mapB
        colors[ns] = (
            "#5E35B1"
            if (inA and inB)
            else ("#1E88E5" if inA else ("#43A047" if inB else "#9E9E9E"))
        )

    return colors


def draw_merged_graph(
    drug_a: int,
    drug_b: int,
    base_dir: str,
    save_path: str,
    resolver: ResolverProtocol,
    highlight_pathway: Optional[str] = None,
    seed: int = 42,
    figsize: Tuple[int, int] = (12, 8),
    size_by_eig: bool = False,
    centrality_box: bool = True,
    interaction_y: Optional[int] = None,
) -> None:
    Ae, An = load_graph_files_cached(drug_a, base_dir)
    Be, Bn = load_graph_files_cached(drug_b, base_dir)
    if Ae is None or Be is None:
        return

    G = merged_graph_cached(Ae, Be, drug_a, drug_b)
    colors = build_node_color_map(
        G,
        An,
        Bn,
        Ae,
        Be,
        drug_a,
        drug_b,
        highlight=[highlight_pathway] if highlight_pathway is not None else None,
    )
    pos = nx.spring_layout(G, seed=seed)

    if size_by_eig:
        try:
            eig = nx.eigenvector_centrality(G, max_iter=EIG_MAX_ITER, tol=EIG_TOL)
            vals_raw = np.array(
                [float(eig.get(n, 0.0)) for n in G.nodes()], dtype=float
            )
            max_val = float(vals_raw.max()) if vals_raw.size > 0 else 0.0
            vals = (
                300.0 + 600.0 * (vals_raw / max_val)
                if max_val > 0.0
                else np.full_like(vals_raw, 500.0)
            )
        except nx.PowerIterationFailedConvergence:
            vals = np.full(len(G.nodes()), 500.0, dtype=float)
    else:
        vals = np.full(len(G.nodes()), 520.0, dtype=float)

    plt.figure(figsize=figsize)
    nx.draw(
        G,
        pos,
        with_labels=True,
        labels={str(n): short_label(n) for n in G.nodes()},
        node_color=[colors[str(n)] for n in G.nodes()],
        edge_color=[d["color"] for _, _, d in G.edges(data=True)],
        node_size=vals,
        font_size=9,
        width=1.1,
    )

    if centrality_box and highlight_pathway is not None and highlight_pathway in G:
        cents = centralities_for_all_nodes_cached(G, drug_a, drug_b)
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
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="#BDBDBD"),
        )

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
    if highlight_pathway is not None:
        ttl += f"\n| highlight: {resolver.pathway_label(highlight_pathway)}"
    if interaction_y is not None:
        ttl += f" | Y={interaction_y}"

    plt.title(ttl, pad=20)
    plt.tight_layout()

    if SAVE_TOGGLE and save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def kde_with_top_markers(
    dist_df: pd.DataFrame,
    pathway: str,
    anchor_a: int,
    interaction_y: int,
    resolver: ResolverProtocol,
    save_path: Optional[str] = None,
    stage: Optional[str] = None,
    metrics: Tuple[str, ...] = METRICS,
) -> None:
    chosen = tuple(m for m in metrics if m in METRICS)
    if len(chosen) == 0:
        return

    n = len(chosen)
    ncols = 2
    nrows = (n + ncols - 1) // ncols

    if len(chosen) == 2:
        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 6))
    else:
        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 5 * nrows))

    axes_arr = np.atleast_1d(axes).ravel()
    for ax in axes_arr[n:]:
        ax.set_visible(False)

    n_int = int((dist_df["group"] == "interacting").sum())
    n_non = int((dist_df["group"] == "non-interacting").sum())
    labels = {
        "interacting": f"interacting (n={n_int})",
        "non-interacting": f"non-interacting (n={n_non})",
    }

    for i, m in enumerate(chosen):
        ax = axes_arr[i]

        for g, color in (("interacting", "red"), ("non-interacting", "blue")):
            sub = _subset_group_df(dist_df, g)
            if sub.empty:
                continue

            sns.kdeplot(
                data=sub,
                x=m,
                fill=True,
                alpha=0.3,
                linewidth=2,
                color=color,
                ax=ax,
                label=labels[g],
            )

        if not dist_df.empty:
            thr = float(dist_df[m].quantile(0.95))
            ymax = float(ax.get_ylim()[1])
            ax.axvline(thr, color="black", linestyle=":", linewidth=1.2)
            ax.text(
                thr,
                ymax * 0.9,
                "95th %",
                rotation=90,
                va="top",
                ha="right",
                fontsize=9,
            )

        ax.set_title(m.capitalize())
        ax.set_xlabel("Centrality")
        ax.set_ylabel("Density")
        ax.legend()

    title = (
        f"A={resolver.drug_label(anchor_a)} | "
        f"Y={interaction_y} | "
        f"Pathway={resolver.pathway_label(pathway)}"
    )
    if stage is not None:
        title = f"[{stage}] " + title

    fig.suptitle(title, fontsize=12)
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))

    if SAVE_TOGGLE and save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def mean_shift_panel(
    dist_df: pd.DataFrame,
    anchor_a: int,
    interaction_y: int,
    pathway: str,
    resolver: ResolverProtocol,
    save_path: Optional[str] = None,
    stage: Optional[str] = None,
) -> None:
    if dist_df.empty:
        return

    d_means: list[float] = []
    ci_los: list[float] = []
    ci_his: list[float] = []

    for m in METRICS:
        a = dist_df.loc[dist_df["group"] == "interacting", m].dropna().to_numpy()
        b = dist_df.loc[dist_df["group"] == "non-interacting", m].dropna().to_numpy()

        dm = (
            float(np.nanmean(a) - np.nanmean(b))
            if (len(a) > 0 and len(b) > 0)
            else np.nan
        )
        lo, hi = bootstrap_mean_diff(a, b, n_boot=BOOT_N, seed=RNG_SEED)
        d_means.append(dm)
        ci_los.append(float(lo))
        ci_his.append(float(hi))

    x = np.arange(len(METRICS))
    plt.figure(figsize=(8, 4.5))
    yerr = [
        np.array(d_means, dtype=float) - np.array(ci_los, dtype=float),
        np.array(ci_his, dtype=float) - np.array(d_means, dtype=float),
    ]
    plt.bar(x, d_means, yerr=yerr, capsize=4, alpha=0.85)
    plt.axhline(0.0, color="#555", linewidth=1)
    plt.xticks(x, [m.capitalize() for m in METRICS])

    ttl = (
        f"Mean diff (int − non) — A={resolver.drug_label(anchor_a)} | "
        f"Y={interaction_y}\n"
        f"Pathway={resolver.pathway_label(pathway)}"
    )
    if stage is not None:
        ttl = f"[{stage}] " + ttl

    plt.title(ttl)
    plt.ylabel("Δ mean")
    plt.tight_layout()

    if SAVE_TOGGLE and save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_volcano_for_metric(
    wdf: pd.DataFrame,
    metric: str,
    out_dir: str,
) -> None:
    ensure_dir(out_dir)
    cols = [f"{metric}_d_mean", f"{metric}_q"]
    if not all(c in wdf.columns for c in cols):
        return

    df = cast(pd.DataFrame, wdf.copy())
    df["effect"] = df[f"{metric}_d_mean"].astype(float)
    df["neglog10q"] = (
        df[f"{metric}_q"]
        .astype(float)
        .map(lambda x: 0.0 if (pd.isna(x) or x <= 0) else -np.log10(x))
    )
    df = cast(
        pd.DataFrame,
        df.sort_values(["neglog10q", "effect"], ascending=[False, False]).reset_index(
            drop=True
        ),
    )

    plt.figure(figsize=(7.5, 5.5))
    plt.scatter(df["effect"], df["neglog10q"], s=16, alpha=0.75)
    plt.axhline(-np.log10(0.05), linestyle="--", linewidth=1.0)
    plt.axvline(0.0, linestyle=":", linewidth=1.0)
    plt.xlabel(f"{metric.capitalize()} Δ mean (int − non)")
    plt.ylabel("−log10(q)")
    plt.title(f"Volcano — {metric} (plain Wilcoxon)")

    if SAVE_TOGGLE:
        plt.savefig(
            os.path.join(out_dir, f"volcano_{metric}.png"),
            dpi=300,
            bbox_inches="tight",
        )
    plt.close()


def rebuild_dist_df_for_combo(
    A: int,
    y: int,
    X: str,
    ddi: pd.DataFrame,
    base_dir: str,
) -> Optional[pd.DataFrame]:
    from cs_analysis_utils.ddi_utils import B_sets_for_label, drug_has_pathway

    B_int, B_non = B_sets_for_label(ddi, A, y)
    B_int_shared = [int(b) for b in B_int if drug_has_pathway(int(b), base_dir, X)]
    B_non_shared = [int(b) for b in B_non if drug_has_pathway(int(b), base_dir, X)]
    if not B_int_shared or not B_non_shared:
        return None

    df_i = measure_X_over_B_fast(A, X, B_int_shared, base_dir, require_shared=True)
    df_n = measure_X_over_B_fast(A, X, B_non_shared, base_dir, require_shared=True)
    if df_i.empty or df_n.empty:
        return None

    df_i = df_i.copy()
    df_n = df_n.copy()
    df_i["group"] = "interacting"
    df_n["group"] = "non-interacting"
    return pd.concat([df_i, df_n], ignore_index=True)


def pick_representative_B_for_graph(
    A: int,
    y: int,
    X: str,
    ddi: pd.DataFrame,
    base_dir: str,
) -> Optional[int]:
    from cs_analysis_utils.ddi_utils import B_sets_for_label, drug_has_pathway

    B_int, _ = B_sets_for_label(ddi, A, y)
    B_int_shared = [int(b) for b in B_int if drug_has_pathway(int(b), base_dir, X)]
    if not B_int_shared:
        return None

    Ae, _ = load_graph_files_cached(A, base_dir)
    if Ae is None:
        return None

    best_b: Optional[int] = None
    best_eig = -np.inf

    for b in B_int_shared:
        Be, _ = load_graph_files_cached(int(b), base_dir)
        if Be is None:
            continue

        G = merged_graph_cached(Ae, Be, A, int(b))
        cents = centralities_for_all_nodes_cached(G, A, int(b))
        eig = float(cents["eigenvector"].get(X, 0.0))
        if eig > best_eig:
            best_eig = eig
            best_b = int(b)

    return best_b if best_b is not None else int(B_int_shared[0])


def visualize_primary_pairs(
    primary_df: pd.DataFrame,
    ddi: pd.DataFrame,
    subg_base: str,
    resolver: ResolverProtocol,
    out_viz_stage_dir: str,
    stage_tag: str,
) -> None:
    if primary_df.empty:
        return

    for _, row in primary_df.iterrows():
        row_s = cast(pd.Series, row)
        A = _row_to_int(row_s, "anchor")
        y = _row_to_int(row_s, "interaction")
        X = _row_to_str(row_s, "pathway")

        anchor_dir = os.path.join(out_viz_stage_dir, f"A{A}")
        ensure_dir(anchor_dir)
        niceX = safe_fname(short_label(X, 60))

        dist_df = rebuild_dist_df_for_combo(A, y, X, ddi, subg_base)
        if dist_df is None:
            continue

        kde_path = os.path.join(anchor_dir, f"KDE_A{A}_Y{y}_X_{niceX}.png")
        kde_with_top_markers(
            dist_df,
            X,
            A,
            y,
            resolver,
            save_path=kde_path,
            stage=stage_tag,
        )

        kde2_path = os.path.join(anchor_dir, f"KDE2_A{A}_Y{y}_X_{niceX}.png")
        kde_with_top_markers(
            dist_df,
            X,
            A,
            y,
            resolver,
            save_path=kde2_path,
            stage=stage_tag,
            metrics=("degree", "eigenvector"),
        )

        meanshift_path = os.path.join(anchor_dir, f"MEANSHIFT_A{A}_Y{y}_X_{niceX}.png")
        mean_shift_panel(
            dist_df,
            A,
            y,
            X,
            resolver,
            save_path=meanshift_path,
            stage=stage_tag,
        )

        b = pick_representative_B_for_graph(A, y, X, ddi, subg_base)
        if b is not None:
            graph_path = os.path.join(
                anchor_dir,
                f"Graph_A{A}_Y{y}_B{b}_X_{niceX}.png",
            )
            draw_merged_graph(
                A,
                b,
                subg_base,
                save_path=graph_path,
                resolver=resolver,
                highlight_pathway=X,
                size_by_eig=False,
                centrality_box=True,
                interaction_y=y,
            )


def visualize_simple_wilcoxon(
    wdf: pd.DataFrame,
    ddi: pd.DataFrame,
    subg_base: str,
    resolver: ResolverProtocol,
    figs_root: str,
) -> None:
    if wdf.empty:
        return

    volcano_dir = os.path.join(figs_root, "volcano")
    ensure_dir(volcano_dir)

    for m in METRICS:
        plot_volcano_for_metric(wdf, m, volcano_dir)

    top_df = cast(
        pd.DataFrame,
        wdf.sort_values(["min_q", "n_p_lt_0_05"], ascending=[True, False])
        .head(20)
        .copy(),
    )

    panels_root = os.path.join(figs_root, "panels")
    ensure_dir(panels_root)

    for _, row in top_df.iterrows():
        row_s = cast(pd.Series, row)
        A = _row_to_int(row_s, "anchor")
        y = _row_to_int(row_s, "interaction")
        X = _row_to_str(row_s, "pathway")

        combo_dir = os.path.join(panels_root, f"A{A}_Y{y}")
        ensure_dir(combo_dir)

        dist_df = rebuild_dist_df_for_combo(A, y, X, ddi, subg_base)
        if dist_df is None:
            continue

        niceX = safe_fname(short_label(X, 60))

        kde_path = os.path.join(combo_dir, f"KDE_A{A}_Y{y}_X_{niceX}.png")
        kde_with_top_markers(
            dist_df,
            X,
            A,
            y,
            resolver,
            save_path=kde_path,
            stage="simple",
        )

        kde2_path = os.path.join(combo_dir, f"KDE2_A{A}_Y{y}_X_{niceX}.png")
        kde_with_top_markers(
            dist_df,
            X,
            A,
            y,
            resolver,
            save_path=kde2_path,
            stage="simple",
            metrics=("degree", "eigenvector"),
        )

        meanshift_path = os.path.join(combo_dir, f"MEANSHIFT_A{A}_Y{y}_X_{niceX}.png")
        mean_shift_panel(
            dist_df,
            A,
            y,
            X,
            resolver,
            save_path=meanshift_path,
            stage="simple",
        )

        b = pick_representative_B_for_graph(A, y, X, ddi, subg_base)
        if b is not None:
            graph_path = os.path.join(
                combo_dir,
                f"Graph_A{A}_Y{y}_B{b}_X_{niceX}.png",
            )
            draw_merged_graph(
                A,
                b,
                subg_base,
                save_path=graph_path,
                resolver=resolver,
                highlight_pathway=X,
                size_by_eig=False,
                centrality_box=True,
                interaction_y=y,
            )
