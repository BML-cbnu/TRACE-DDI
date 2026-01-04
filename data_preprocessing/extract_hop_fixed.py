import argparse
import os
from typing import List, Optional, Tuple

import networkx as nx
import pandas as pd
from tqdm import tqdm

pd.set_option("mode.chained_assignment", None)


# =========================
# I/O utilities
# =========================
def load_nodes_edges(
    nodes_base_path: str, edges_base_path: str, compound_num: int
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[str]]:
    """Load per-compound node/edge TSV files with safety checks."""

    nodes_path = os.path.join(nodes_base_path, f"compound{compound_num}_nodes.tsv")
    edges_path = os.path.join(edges_base_path, f"compound{compound_num}_edges.tsv")

    missing = []
    if not os.path.exists(nodes_path):
        missing.append("nodes")
    if not os.path.exists(edges_path):
        missing.append("edges")

    if missing:
        return None, None, f"missing_{'+'.join(missing)}"

    nodes = pd.read_csv(
        nodes_path,
        sep="\t",
        names=["node_num", "node_name", "node_type", "spread_value", "visted_count"],
    )
    edges = pd.read_csv(
        edges_path,
        sep="\t",
        names=["head", "tail", "relation"],
    )

    if nodes.empty:
        return None, None, "empty_nodes"
    if edges.empty:
        return None, None, "empty_edges"

    return nodes, edges, None


# =========================
# Graph construction
# =========================
def build_digraph(edges: pd.DataFrame) -> nx.DiGraph:
    """Build directed graph from edge list."""
    G = nx.DiGraph()
    for h, t, r in edges[["head", "tail", "relation"]].itertuples(
        index=False, name=None
    ):
        G.add_edge(h, t, relation=r)
    return G


# =========================
# Path search
# =========================
def find_paths_within_k(
    G: nx.DiGraph, source: int, targets: List[int], num_hop: int
) -> List[List[int]]:
    """Find shortest paths within k hops (edge count)."""
    paths = []
    for target in targets:
        try:
            path = nx.shortest_path(G, source=source, target=target)
            if (len(path) - 1) <= num_hop:
                paths.append(path)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
    return paths


# =========================
# Meta-edge extraction
# =========================
def make_meta_edges(edges: pd.DataFrame, paths: List[List[int]]) -> pd.DataFrame:
    """Extract unique edges appearing in meta-paths."""
    meta_edges = set()

    for path in paths:
        for u, v in zip(path[:-1], path[1:]):
            rows = edges[(edges["head"] == u) & (edges["tail"] == v)][
                ["head", "relation", "tail"]
            ]
            for h, r, t in rows.itertuples(index=False, name=None):
                meta_edges.add((int(h), str(r), int(t)))

    if not meta_edges:
        return pd.DataFrame(columns=["head", "relation", "tail"])

    df = pd.DataFrame(sorted(meta_edges), columns=["head", "relation", "tail"])
    return df.drop_duplicates(subset=["head", "relation", "tail"])


# =========================
# Main pipeline
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prob", type=float, default=0.3)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--num_hop", type=int, default=4)
    parser.add_argument("--nodes_type", type=str, default="CGPD")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=1705)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # -------------------------
    # Base data directories
    # -------------------------
    base_data_dir = "/path/to/your/data"

    nodes_base_path = (
        f"{base_data_dir}/{args.nodes_type}/rw_mean/"
        f"steps_{args.steps}/prob_{args.prob}/nodes"
    )
    edges_base_path = (
        f"{base_data_dir}/{args.nodes_type}/rw_mean/"
        f"steps_{args.steps}/prob_{args.prob}/edges"
    )

    # -------------------------
    # Output directories
    # -------------------------
    out_edges = (
        f"{base_data_dir}/{args.nodes_type}/rw_mean/"
        f"steps_{args.steps}/prob_{args.prob}/hop{args.num_hop}/edges"
    )
    out_nodes = (
        f"{base_data_dir}/{args.nodes_type}/rw_mean/"
        f"steps_{args.steps}/prob_{args.prob}/hop{args.num_hop}/nodes"
    )

    os.makedirs(out_edges, exist_ok=True)
    os.makedirs(out_nodes, exist_ok=True)

    # -------------------------
    # Diagnostics
    # -------------------------
    missing = {}
    no_pathway = []
    no_metapath = []
    source_missing = []
    saved = []

    for compound_num in tqdm(
        range(args.start, args.end + 1), desc="Processing compounds"
    ):
        nodes, edges, reason = load_nodes_edges(
            nodes_base_path, edges_base_path, compound_num
        )
        if reason:
            missing[compound_num] = reason
            continue

        G = build_digraph(edges)

        # Skip if source node not in graph
        if compound_num not in G:
            source_missing.append(compound_num)
            continue

        # Select pathway nodes
        pathway_nodes = nodes[nodes["node_name"].astype(str).str.startswith("Pathway")][
            "node_num"
        ].tolist()

        if not pathway_nodes:
            no_pathway.append(compound_num)
            continue

        # Find k-hop meta-paths
        paths = find_paths_within_k(G, compound_num, pathway_nodes, args.num_hop)

        if not paths:
            no_metapath.append(compound_num)
            continue

        # Extract meta-graph
        meta_edges = make_meta_edges(edges, paths)
        if meta_edges.empty:
            no_metapath.append(compound_num)
            continue

        meta_nodes = nodes[
            nodes["node_num"].isin(meta_edges["head"])
            | nodes["node_num"].isin(meta_edges["tail"])
        ]

        # Save results
        meta_edges.to_csv(
            os.path.join(out_edges, f"compound{compound_num}_edges.tsv"),
            sep="\t",
            index=False,
            header=False,
        )
        meta_nodes.to_csv(
            os.path.join(out_nodes, f"compound{compound_num}_nodes.tsv"),
            sep="\t",
            index=False,
            header=False,
        )

        saved.append(compound_num)

    # -------------------------
    # Summary
    # -------------------------
    print("\n--- Process Completed ---")
    print(f"Saved OK: {len(saved)}")
    print(f"Missing/empty input: {len(missing)}")
    print(f"Source not in graph: {len(source_missing)}")
    print(f"No pathway: {len(no_pathway)}")
    print(f"No meta-path: {len(no_metapath)}")


if __name__ == "__main__":
    main()
