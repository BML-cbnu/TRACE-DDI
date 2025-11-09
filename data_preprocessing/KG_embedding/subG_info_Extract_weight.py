import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import argparse


class SubCon2d(nn.Module):
    """2D convolution-based subgraph encoder"""
    def __init__(self, conv_out_dim=128, final_vector_length=20):
        super(SubCon2d, self).__init__()
        self.conv2d = nn.Conv2d(
            in_channels=1, out_channels=conv_out_dim,
            kernel_size=(5, 5), stride=(3, 3), padding=(1, 1)
        )
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(conv_out_dim, final_vector_length)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class SubCon1d(nn.Module):
    """1D convolution layer to reduce feature dimensionality"""
    def __init__(self, input_length, output_length):
        super(SubCon1d, self).__init__()
        filter_size = output_length
        stride = max((input_length - filter_size) // (output_length - 1), 1)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=filter_size, stride=stride, padding=0)

    def forward(self, x):
        x = x.unsqueeze(0).unsqueeze(0)
        x = self.conv1(x)
        return x.squeeze()


def process_single_compound(i, base_dir, entity_emb, rel_emb, vector_length, method, n_components=5):
    """
    Process a single compound: load subgraph data, compute compound vector using the given method.
    """
    edges_file = os.path.join(base_dir, f'edges/compound{i}_edges.tsv')
    nodes_file = os.path.join(base_dir, f'nodes/compound{i}_nodes.tsv')

    if not os.path.exists(edges_file) or not os.path.exists(nodes_file):
        return None

    try:
        edges_df = pd.read_csv(
            edges_file, sep='\t', header=None,
            names=['node1', 'relation_id', 'node2', 'ori_node1_num', 'ori_node2_num', 'node1_name', 'node2_name']
        )
        nodes_df = pd.read_csv(
            nodes_file, sep='\t', header=None,
            names=['node_id', 'node_name', 'node_type', 'visited_count', 'spread_value', 'ori_node_id']
        )

        # Extract entities and relations
        subgraph_entity_ids = list(set(edges_df['ori_node1_num'].tolist() + edges_df['ori_node2_num'].tolist()))
        subgraph_relation_ids = list(set(edges_df['relation_id'].tolist()))
        subgraph_entity_emb = entity_emb[subgraph_entity_ids]
        subgraph_relation_emb = rel_emb[subgraph_relation_ids]

        # Scale node attributes
        scaler = MinMaxScaler()
        nodes_df[['spread_value', 'visited_count']] = scaler.fit_transform(nodes_df[['spread_value', 'visited_count']])

        # Apply node weights (for drug nodes only)
        for idx, node_id in enumerate(subgraph_entity_ids):
            node_info = nodes_df[nodes_df['ori_node_id'] == node_id]
            if not node_info.empty and node_info.iloc[0]['node_type'] == 9:
                spread_value = node_info.iloc[0]['spread_value']
                visited_count = node_info.iloc[0]['visited_count']
                weight = (spread_value + visited_count) / 2
                subgraph_entity_emb[idx] *= weight

        combined_emb = np.concatenate((subgraph_entity_emb, subgraph_relation_emb), axis=0)

        # Method-based embedding generation
        if method == 'pca':
            pca = PCA(n_components=min(n_components, combined_emb.shape[1]))
            compound_vector = pca.fit_transform(combined_emb).flatten()

        elif method == 'sum':
            compound_vector = np.sum(combined_emb, axis=0)
            conv_layer = SubCon1d(len(compound_vector), vector_length)
            compound_vector = conv_layer(torch.tensor(compound_vector, dtype=torch.float32)).detach().numpy()

        elif method == 'mean':
            compound_vector = np.mean(combined_emb, axis=0)
            conv_layer = SubCon1d(len(compound_vector), vector_length)
            compound_vector = conv_layer(torch.tensor(compound_vector, dtype=torch.float32)).detach().numpy()

        elif method == 'product':
            compound_vector = np.prod(combined_emb, axis=0)
            conv_layer = SubCon1d(len(compound_vector), vector_length)
            compound_vector = conv_layer(torch.tensor(compound_vector, dtype=torch.float32)).detach().numpy()

        elif method == 'conv':
            combined_emb_tensor = torch.tensor(combined_emb, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            processor = SubCon2d(conv_out_dim=128, final_vector_length=vector_length)
            compound_vector = processor(combined_emb_tensor).detach().numpy().flatten()

        else:
            raise ValueError(f"Unknown method: {method}")

        # Adjust vector length
        if len(compound_vector) < vector_length:
            compound_vector = np.pad(compound_vector, (0, vector_length - len(compound_vector)))
        else:
            compound_vector = compound_vector[:vector_length]

        return compound_vector

    except Exception as e:
        print(f"Error processing compound {i}: {e}")
        return None


def generate_compound_vectors(base_dir, embed_dir, vector_length=20, method='conv', save_dir='./data'):
    """
    Main routine to process all compounds and save their vectors.
    """
    print(f"Starting compound vector generation ({method}, length={vector_length})")

    entity_emb = np.load(os.path.join(embed_dir, 'DRKG_TransE_l2_entity.npy'))
    rel_emb = np.load(os.path.join(embed_dir, 'DRKG_TransE_l2_relation.npy'))

    compound_vectors = {}
    for i in tqdm(range(1705 + 1)):
        vector = process_single_compound(i, base_dir, entity_emb, rel_emb, vector_length, method)
        if vector is not None:
            compound_vectors[i] = vector

    compound_vectors_df = pd.DataFrame.from_dict(compound_vectors, orient='index')
    output_file = os.path.join(save_dir, f'vec_{vector_length}_{method}.csv')
    compound_vectors_df.to_csv(output_file, index_label='compound_id')

    print(f"Compound vectors saved to: {output_file}")


parser = argparse.ArgumentParser(description='Generate compound-level vectors from subgraph data.')
parser.add_argument('--base_dir', type=str, default='./data/CGPD/subG_modify', help='Path to the modified subgraph directory')
parser.add_argument('--embed_dir', type=str, default='./data/drkg/embed',help='Path to embedding files (entity/relation npy)')
parser.add_argument('--save_dir', type=str, default='./data/CGPD',help='Output directory for compound vector CSV')
parser.add_argument('--vector_length', type=int, default=20, help='Final compound vector length')
parser.add_argument('--method', type=str, default='conv', choices=['pca', 'sum', 'mean', 'product', 'conv'], help='Vector generation method')
args = parser.parse_args()

for arg, value in vars(args).items():
    print(f"{arg}: {value}")

generate_compound_vectors(
    base_dir=args.base_dir,
    embed_dir=args.embed_dir,
    vector_length=args.vector_length,
    method=args.method,
    save_dir=args.save_dir)
