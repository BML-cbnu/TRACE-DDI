import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import os
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import argparse
import torch
import torch.nn as nn

class subCon(nn.Module):
    def __init__(self, conv_out_dim, final_vector_length):
        super(subCon, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=1, out_channels=conv_out_dim, kernel_size=(5, 5), stride=(3, 3), padding=(1, 1))
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(conv_out_dim, final_vector_length)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class conV1D(nn.Module):
    def __init__(self, input_length, output_length):
        super(conV1D, self).__init__()
        filter_size = output_length
        stride = (input_length - filter_size) // (output_length - 1)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=filter_size, stride=stride, padding=0)

    def forward(self, x):
        x = x.unsqueeze(0).unsqueeze(0)
        x = self.conv1(x)
        return x.squeeze()

parser = argparse.ArgumentParser(description='Process compound vectors with different methods.')
parser.add_argument('--vector_length', type=int, default=20)
parser.add_argument('--method', type=str, default='conv', choices=['pca', 'sum', 'mean', 'product', 'conv'], help='Method to process the compound vectors.')

args = parser.parse_args()

vector_length = args.vector_length
n_components = 5
method = args.method

entity_emb = np.load('/path/to/your/data/drkg/embed/DRKG_TransE_l2_entity.npy')
rel_emb = np.load('/path/to/your/data/drkg/embed/DRKG_TransE_l2_relation.npy')
compound_vectors_df = pd.DataFrame()

base_dir = '/path/to/your/data/CGPD/subG_modify/'

for i in tqdm(range(1706)):
    edges_file = os.path.join(base_dir, f'edges/compound{i}_edges.tsv')
    nodes_file = os.path.join(base_dir, f'nodes/compound{i}_nodes.tsv')
    if not os.path.exists(edges_file) or not os.path.exists(nodes_file):
        continue

    try:
        edges_df = pd.read_csv(edges_file, sep='\t', header=None, names=['node1', 'relation_id', 'node2', 'ori_node1_num', 'ori_node2_num', 'node1_name', 'node2_name'])
        nodes_df = pd.read_csv(nodes_file, sep='\t', header=None, names=['node_id', 'node_name', 'node_type', 'visited_count', 'spread_value', 'ori_nude_id'])

        subgraph_entity_ids = list(set(edges_df['ori_node1_num'].tolist() + edges_df['ori_node2_num'].tolist()))
        subgraph_relation_ids = list(set(edges_df['relation_id'].tolist()))

        subgraph_entity_emb = entity_emb[subgraph_entity_ids]
        subgraph_relation_emb = rel_emb[subgraph_relation_ids]

        scaler = MinMaxScaler()
        nodes_df[['spread_value', 'visited_count']] = scaler.fit_transform(nodes_df[['spread_value', 'visited_count']])
        
        for idx, node_id in enumerate(subgraph_entity_ids):
            node_info = nodes_df[nodes_df['ori_nude_id'] == node_id]
            if not node_info.empty and node_info.iloc[0]['node_type'] == 9:
                spread_value = node_info.iloc[0]['spread_value']
                visited_count = node_info.iloc[0]['visited_count']
                weight = (spread_value + visited_count) / 2
                subgraph_entity_emb[idx] *= weight

        combined_emb = np.concatenate((subgraph_entity_emb, subgraph_relation_emb), axis=0)
       
        if method == 'pca':
            pca = PCA(n_components=n_components)
            compound_vector = pca.fit_transform(combined_emb)
            compound_vector = compound_vector.flatten()
            
            if len(compound_vector) < vector_length:
                compound_vector = np.pad(compound_vector, (0, vector_length - len(compound_vector)), 'constant')
            else:
                compound_vector = compound_vector[:vector_length]
            
        elif method == 'sum':
            compound_vector = np.sum(combined_emb, axis=0)
            conv_layer = conV1D(400, vector_length)
            compound_vector = conv_layer(torch.tensor(compound_vector, dtype=torch.float32)).detach().numpy()

        elif method == 'mean':
            compound_vector = np.mean(combined_emb, axis=0)
            conv_layer = conV1D(400, vector_length)
            compound_vector = conv_layer(torch.tensor(compound_vector, dtype=torch.float32)).detach().numpy()
        
        elif method == 'product':
            compound_vector = np.prod(combined_emb, axis=0)
            conv_layer = conV1D(400, vector_length)
            compound_vector = conv_layer(torch.tensor(compound_vector, dtype=torch.float32)).detach().numpy()
        
        elif method == 'conv':
            combined_emb_tensor = torch.tensor(combined_emb, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            processor = subCon(conv_out_dim=128, final_vector_length=vector_length)
            compound_vector = processor(combined_emb_tensor).detach().numpy().flatten()

        compound_df = pd.DataFrame([compound_vector], index=[i])
        compound_vectors_df = pd.concat([compound_vectors_df, compound_df])

    except Exception as e:
        print(f"Error processing compound {i}: {e}")

try:
    output_file = '/path/to/your/data/vec{}_{}.csv'.format(vector_length, method)
    compound_vectors_df.to_csv(output_file, index=True)  
    print(f"CSV file successfully saved to {output_file}")
except Exception as e:
    print(f"Error saving CSV file: {e}")
