# ---------- Imports and Settings ----------
import pandas as pd
import networkx as nx
import os
import numpy as np
import pickle
import argparse
from multiprocessing import Pool, Manager
import logging

pd.set_option('mode.chained_assignment', None)
logging.basicConfig(level=logging.INFO)

# ---------- Argument Parsing ----------
parser = argparse.ArgumentParser()
parser.add_argument('--prob', type=float, default=0.3)  # RWR restart probability
parser.add_argument('--steps', type=int, default=2)  # RWR steps
parser.add_argument('--iteration', type=int, default=1)  # Iterations per compound
parser.add_argument('--num_workers', type=int, default=50)  # Multiprocessing workers
parser.add_argument('--nodes_name', type=str, default='CGPD')  # Node set identifier
args = parser.parse_args()

# ---------- Paths ----------
base_dir = f'/home/your/project/data/{args.nodes_name}/'
nodes_file_path = os.path.join(base_dir, f'nodes_{args.nodes_name}.tsv')
edges_file_path = os.path.join(base_dir, f'edges_{args.nodes_name}.tsv')
graph_file_path = os.path.join(base_dir, 'graph_drkg.pkl')
subgraph_file_path = os.path.join(base_dir, f'subgraph_{args.nodes_name}.pkl')
drkg_data_dir = "/home/your/project/data/drkg"


# ---------- Step 1: Extract Nodes and Edges ----------
def extract_nodes_and_edges(nodes_type, nodes_name):
    """Filter nodes and edges based on selected types and save as TSV."""
    os.makedirs(base_dir, exist_ok=True)
    nodes_filename = os.path.join(base_dir, f"nodes_{nodes_name}.tsv")
    edges_filename = os.path.join(base_dir, f"edges_{nodes_name}.tsv")

    kg_nodes = pd.read_csv(os.path.join(drkg_data_dir, "nodes.tsv"), sep='\t',
                           names=['node_num','name','node_embed'])
    kg_edges = pd.read_csv(os.path.join(drkg_data_dir, "edges.tsv"), sep='\t',
                           names=['head','relation','tail'])

    # Filter nodes and edges
    pre_nodes = kg_nodes[kg_nodes['node_embed'].isin(nodes_type)]
    pre_edges = kg_edges[
        (kg_edges['head'].isin(pre_nodes['node_num'])) &
        (kg_edges['tail'].isin(pre_nodes['node_num']))
    ]

    # Save filtered files
    pre_nodes.to_csv(nodes_filename, sep='\t', index=False, header=False)
    pre_edges.to_csv(edges_filename, sep='\t', index=False, header=False)
    logging.info(f"Filtered nodes and edges saved to {base_dir}")


# ---------- Step 2: Load Data ----------
def load_data():
    """Load filtered nodes and edges."""
    nodes_df = pd.read_csv(nodes_file_path, sep='\t', names=['node_num','node_name','node_type'])
    edges_df = pd.read_csv(edges_file_path, sep='\t', names=['head','relation','tail'])
    return nodes_df, edges_df


def load_or_create_graph(edges_list):
    """Load or build full graph."""
    if os.path.exists(graph_file_path):
        with open(graph_file_path, 'rb') as f:
            return pickle.load(f)
    kg_nx = nx.DiGraph()
    for h,r,t in edges_list:
        kg_nx.add_edge(h,t,relation=r)
    with open(graph_file_path,'wb') as f:
        pickle.dump(kg_nx,f)
    return kg_nx


def load_or_create_subgraph(graph, nodes_df):
    """Load or build subgraph containing selected nodes."""
    if os.path.exists(subgraph_file_path):
        with open(subgraph_file_path,'rb') as f:
            return pickle.load(f)
    sub_nodes = nodes_df['node_num'].values
    sub_kg = graph.subgraph(sub_nodes).copy()
    
    with open(subgraph_file_path,'wb') as f:
        pickle.dump(sub_kg,f)
    return sub_kg





# ---------- Step 3: Random Walk Process ----------
def random_walk_process(args):
    """Perform Random Walk with Restart for a single compound and save subgraph."""
    compound, sub_kg, nodes_df, pathway_nodes, restart_prob, max_steps, n_iter, not_pathway = args
    sum_values = {n:0 for n in sub_kg.nodes}
    sum_visited = {n:0 for n in sub_kg.nodes}

    for _ in range(n_iter):
        values = {n:0 for n in sub_kg.nodes}
        visited = {n:0 for n in sub_kg.nodes}
        values[compound] = 1e6
        current = compound

        for i in range(1, max_steps+1):
            neighbors = list(nx.classes.function.all_neighbors(sub_kg_nx, current))
            visited[current] += 1
            if neighbors:
                transfer = values[current]/(len(neighbors)*i)
                for n in neighbors:
                    values[n] += transfer
                values[current] -= transfer*len(neighbors)

                # Random restart
                current = np.random.choice(neighbors) if np.random.rand() > restart_prob else compound

        # Accumulate results
        for n in sum_values:
            sum_values[n] += values[n]
            sum_visited[n] += visited[n]

    # Average over iterations
    avg_values = {n: sum_values[n]/n_iter for n in sum_values}
    avg_visited = {n: sum_visited[n]/n_iter for n in sum_visited}

    # Save subgraph and node info
    visited_subgraph = sub_kg.subgraph([n for n in avg_visited if avg_visited[n]>0])
    nodes_info = nodes_df[nodes_df['node_num'].isin(visited_subgraph.nodes)].copy()
    nodes_info['value'] = nodes_info['node_num'].map(avg_values)
    nodes_info['visited_counts'] = nodes_info['node_num'].map(avg_visited)

    if not any(nodes_info['node_num'].isin(pathway_nodes)):
        not_pathway.append(compound)

    nodes_dir = f"{base_dir}/rw_mean/steps_{max_steps}/prob_{restart_prob}/nodes"
    edges_dir = f"{base_dir}/rw_mean/steps_{max_steps}/prob_{restart_prob}/edges"
    os.makedirs(nodes_dir, exist_ok=True)
    os.makedirs(edges_dir, exist_ok=True)

    nodes_info.to_csv(f"{nodes_dir}/compound{compound}_nodes.tsv", sep='\t', index=False, header=False)
    nx.write_edgelist(visited_subgraph, f"{edges_dir}/compound{compound}_edges.tsv", data=['relation'], delimiter='\t')
    logging.info(f"Compound {compound} RWR result saved.")


# ---------- Main ----------
if __name__ == '__main__':
    
    ## Compound: 0, Anatomy: 1, Atc: 2, Process: 3, Cellular Component: 4, Compound: 5, Disease: 6
    ## Gene: 7, Molecular Function: 8, Pathway: 9
    ##Pharmacologic: 10, Side Effect: 11, Symptom: 12

    nodes_type_mapping = {
    'CGPD': [0,5,7,9,6],  # Compound, Gene, Pathway, Disease
    'CGPDS': [0,5,7,9,6, 11] # Compound, Gene, Pathway, Disease, Side-Effect
    }
    
    if args.nodes_name in nodes_type_mapping:
        nodes_type = nodes_type_mapping[args.nodes_name]
    else:
        raise ValueError(f"nodes_name {args.nodes_name} not found in nodes_type_mapping. Please provide --nodes_type explicitly.")
    
    extract_nodes_and_edges(nodes_type, args.nodes_name)  # Step 1
    nodes_df, edges_df = load_data()  # Step 2
    edges_list = list(zip(edges_df['head'], edges_df['relation'], edges_df['tail']))
    edges_list.extend([(1603, 1603, 1603), (1612, 1612, 1612), (1623, 1623, 1623), # Adding self-loops ensures these nodes exist in the graph
                    (1627, 1627, 1627), (1653, 1653, 1653), (1696, 1696, 1696)])

    graph = load_or_create_graph(edges_list)  # Load/build full graph
    sub_kg_nx = load_or_create_subgraph(graph, nodes_df)  # Load/build subgraph
    
    compound_nodes = nodes_df[nodes_df['node_type']==0]['node_num'].tolist()  # Compound seeds
    pathway_nodes = nodes_df[nodes_df['node_type']==9]['node_num'].tolist()  # Pathway nodes

    # Step 3: Run RWR with multiprocessing
    with Manager() as manager:
        not_pathway = manager.list()
        args_list = [(c, sub_kg_nx, nodes_df, pathway_nodes, args.prob, args.steps, args.iteration, not_pathway)
                     for c in compound_nodes]

        logging.info("Starting multiprocessing pool...")
        from multiprocessing import Pool
        with Pool(args.num_workers) as pool:
            pool.map(random_walk_process, args_list)

        logging.info(f'Total compounds not reaching any pathway > {len(not_pathway)}: {list(not_pathway)}')