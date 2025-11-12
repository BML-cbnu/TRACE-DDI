import pandas as pd
import networkx as nx
import os
from tqdm import tqdm
import argparse

pd.set_option('mode.chained_assignment', None)

# --- Argument Parsing --- #
parser = argparse.ArgumentParser()
parser.add_argument('--prob', type=float, default=0.5)
parser.add_argument('--steps', type=int, default=2)
parser.add_argument('--num_hop', type=int, default=5)
parser.add_argument('--nodes_type', type=str, default='CGPD')
args = parser.parse_args()

# Params
steps = args.steps
prob = args.prob
num_hop = args.num_hop
nodes_type = args.nodes_type

# Base directories
nodes_base_path = f'/home/your/project/data/{nodes_type}/rw_mean/steps_{steps}/prob_{prob}/nodes'
edges_base_path = f'/home/your/project/data/{nodes_type}/rw_mean/steps_{steps}/prob_{prob}/edges'

# Output directories — created once
output_edges_dir = f'/home/your/project/data/{nodes_type}/rw_mean/steps_{steps}/prob_{prob}/hop{num_hop}/edges'
output_nodes_dir = f'/home/your/project/data/{nodes_type}/rw_mean/steps_{steps}/prob_{prob}/hop{num_hop}/nodes'

os.makedirs(output_edges_dir, exist_ok=True)
os.makedirs(output_nodes_dir, exist_ok=True)

# Missing info tracking
not_pathway_count = []
not_metapath_count = []

# Main loop over compounds (example: 0 to 1705)
for compound_num in tqdm(range(1706), desc="Processing compounds"):

    nodes_file_path = os.path.join(nodes_base_path, f'compound{compound_num}_nodes.tsv')
    edges_file_path = os.path.join(edges_base_path, f'compound{compound_num}_edges.tsv')

    if not os.path.exists(nodes_file_path) or not os.path.exists(edges_file_path):
        print(f"[WARN] Files for compound {compound_num} not found. Skipping...")
        continue

    nodes = pd.read_csv(nodes_file_path, sep='\t', names=['node_num', 'node_name', 'node_type', 'spread_value', 'visited_count'])
    edges = pd.read_csv(edges_file_path, sep='\t', names=['head', 'relation', 'tail'])

    # Create directed graph
    kg_nx = nx.DiGraph()
    for _, row in edges.iterrows():
        kg_nx.add_edge(row['head'], row['tail'], relation=row['relation'])

    # Extract Pathway nodes from nodes file
    pathways = nodes[nodes['node_name'].str.startswith('Pathway')]['node_num'].tolist()

    if not pathways:
        not_pathway_count.append(compound_num)
        not_metapath_count.append(compound_num)
        continue

    print(f'Compound {compound_num} → Searching pathways (steps={steps}, prob={prob}, num_hop={num_hop})...')
    meta_paths = []

    for pathway in pathways:
        try:
            path = nx.shortest_path(kg_nx, source=compound_num, target=pathway)
            if len(path) <= (num_hop + 1):  # Node count = hop + 1
                meta_paths.append(path)
        except nx.NetworkXNoPath:
            continue

    if not meta_paths:
        print(f'No meta-path found for compound {compound_num} within {num_hop} hops.')
        not_metapath_count.append(compound_num)
        continue

    # Save edges from meta-paths
    edge_path = os.path.join(output_edges_dir, f'compound{compound_num}_edges.tsv')
    with open(edge_path, 'w') as file:
        for path in meta_paths:
            for i in range(len(path) - 1):
                edge_data = edges[(edges['head'] == path[i]) & (edges['tail'] == path[i + 1])]
                for _, ed in edge_data.iterrows():
                    file.write(f"{ed['head']}\t{ed['relation']}\t{ed['tail']}\n")

    # Save corresponding nodes
    meta_edges = pd.read_csv(edge_path, sep='\t', names=['head', 'relation', 'tail'])
    meta_nodes = nodes[nodes['node_num'].isin(meta_edges['head']) | nodes['node_num'].isin(meta_edges['tail'])]
    meta_nodes.to_csv(os.path.join(output_nodes_dir, f'compound{compound_num}_nodes.tsv'), sep='\t', index=False, header=False)

# Final report
print(f"\n--- Process Completed ---")
print(f"Total compounds processed: 1706")
print(f"Missing pathway info: {len(not_pathway_count)} compounds")
print(f"Missing meta-path: {len(not_metapath_count)} compounds")
print(f"Missing pathway compounds: {not_pathway_count}")
print(f"Missing meta-path compounds: {not_metapath_count}")