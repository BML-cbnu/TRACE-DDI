import argparse
import os

import networkx as nx
import pandas as pd
from tqdm import tqdm

pd.set_option("mode.chained_assignment", None)


# --- Argument Parsing --- #
parser = argparse.ArgumentParser()
parser.add_argument("--prob", type=float, default=0.3)
parser.add_argument("--steps", type=int, default=20000)
parser.add_argument("--num_hop", type=int, default=4)
parser.add_argument("--nodes_type", type=str, default="CGPD")
args = parser.parse_args()

# Params
steps = args.steps
prob = args.prob
num_hop = args.num_hop
nodes_type = args.nodes_type

# Base directories
nodes_base_path = (
    f"/path/to/your/project/data/{nodes_type}/rw_mean/steps_{steps}/prob_{prob}/nodes"
)
edges_base_path = (
    f"/path/to/your/project/data/{nodes_type}/rw_mean/steps_{steps}/prob_{prob}/edges"
)

# Output directories — created once
output_edges_dir = f"/path/to/your/project/data/{nodes_type}/rw_mean/steps_{steps}/prob_{prob}/hop{num_hop}/edges"
output_nodes_dir = f"/path/to/your/project/data/{nodes_type}/rw_mean/steps_{steps}/prob_{prob}/hop{num_hop}/nodes"


def load_nodes_edges(nodes_base_path, edges_base_path, compound_num):
    nodes_file_path = os.path.join(nodes_base_path, f"compound{compound_num}_nodes.tsv")
    edges_fil_path = os.path.join(edges_base_path, f"compound{compound_num}_edges.tsv")

    if not (os.path.exists(nodes_file_path) and os.path.exists(edges_fil_path)):
        return None, None

    nodes = pd.read_csv(
        nodes_file_path,
        sep="\t",
        names=["node_num", "node_name", "node_type", "spread_value", "visted_count"],
    )
    edges = pd.read_csv(edges_fil_path, sep="\t", names=["head", "tail", "relation"])
    return nodes, edges


def build_digraph(edge_list):
    G = nx.DiGraph()
    for head, relation, tail in edge_list:
        G.add_edge(head, tail, relation=relation)
    return G


def meta_paths(G, source, target_list, num_hop):
    meta_paths = []
    for target in target_list:
        try:
            path = nx.shortest_path(G, source=source, target=target)
            if len(path) <= (num_hop + 1):
                meta_paths.append(path)
        except nx.NetworkXNoPath:
            continue
        except nx.NodeNotFound:
            continue
    return meta_paths


def make_meta_edges(edges, meta_paths):
    meta_edges = set()  # (head, relation, tail)

    for path in meta_paths:
        for u, v in zip(path[:-1], path[1:]):
            rows = edges[(edges["head"] == u) & (edges["tail"] == v)][
                ["head", "relation", "tail"]
            ]
            for h, r, t in rows.itertuples(index=False):
                meta_edges.add((int(h), str(r), int(t)))
    meta_edges = pd.DataFrame(sorted(meta_edges), columns=["head", "relation", "tail"])
    meta_edges = meta_edges.drop_duplicates(
        subset=["head", "relation", "tail"], keep="first"
    )
    return meta_edges


if __name__ == "__main__":
    os.makedirs(output_edges_dir, exist_ok=True)
    os.makedirs(output_nodes_dir, exist_ok=True)

    # Missing info tracking
    not_pathway_count = []
    not_metapath_count = []

    for compound_num in tqdm(range(0, 1705 + 1), desc="Processing compounds"):
        nodes, edges = load_nodes_edges(nodes_base_path, edges_base_path, compound_num)

        # Create directed graph
        edge_list = list(zip(edges["head"], edges["relation"], edges["tail"]))
        kg_nx = build_digraph(edge_list)

        pathway_list = nodes[nodes["node_name"].str.startswith("Pathway")][
            "node_num"
        ].tolist()  # pathway_list

        # case 1: Pathway entities are not included for compound in subgraph(random walk)
        if not pathway_list:
            not_pathway_count.append(compound_num)
            not_metapath_count.append(compound_num)
            continue

        print(
            f"Compound {compound_num} → Searching pathways (steps={steps}, prob={prob}, num_hop={num_hop})..."
        )
        mata_paths = meta_paths(
            kg_nx, source=compound_num, target_list=pathway_list, num_hop=num_hop
        )

        # case 2: metapath is not extracted for compound within k-hops.
        if not mata_paths:
            print(
                f"No meta-path found for compound {compound_num} within {num_hop} hops."
            )
            not_metapath_count.append(compound_num)
            continue

        meta_edges = make_meta_edges(edges, mata_paths)
        meta_nodes = nodes[
            nodes["node_num"].isin(meta_edges["head"])
            | nodes["node_num"].isin(meta_edges["tail"])
        ]
        meta_edges.to_csv(
            os.path.join(output_edges_dir, f"compound{compound_num}_edges.tsv"),
            sep="\t",
            index=False,
            header=False,
        )
        meta_nodes.to_csv(
            os.path.join(output_nodes_dir, f"compound{compound_num}_nodes.tsv"),
            sep="\t",
            index=False,
            header=False,
        )

    # Final report
    print(f"\n--- Process Completed ---")
    print(f"Total compounds processed: 1706(0-1705)")
    print(f"Missing pathway info: {len(not_pathway_count)} compounds")
    print(f"Missing meta-path: {len(not_metapath_count)} compounds")
    print(f"Missing pathway compounds: {not_pathway_count}")
    print(f"Missing meta-path compounds: {not_metapath_count}")
