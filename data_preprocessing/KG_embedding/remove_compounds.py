import pandas as pd
import os

def delete_files(numbers, base_edge_path, base_node_path):
    for number in numbers:
        edge_file = base_edge_path.format(number)
        node_file = base_node_path.format(number)
        
        if os.path.exists(edge_file):
            os.remove(edge_file)
            print(f'Deleted: {edge_file}')
        else:
            print(f'File not found: {edge_file}')
        
        if os.path.exists(node_file):
            os.remove(node_file)
            print(f'Deleted: {node_file}')
        else:
            print(f'File not found: {node_file}')


ddi_file = '/path/to/your/data/DDI/ddi.tsv'
ddi_output_file = '/path/to/your/data/DDI/ddi_01.tsv'
smiles_file = '/path/to/your/data/DDI/smiles.tsv'
smiles_output_file = '/path/to/your/data/DDI/smiles_01.tsv'


ddi_df = pd.read_csv(ddi_file, sep='\t', names=['drug1', 'drug2', 'interaction_type'])
smiles_df = pd.read_csv(smiles_file, sep='\t', header=None, names=['compound', 'smiles'])
compounds_to_remove = [651, 1170, 1359, 1475, 1509, 1603, 1612, 1623, 1627, 1634, 1653, 1684, 1696, 1698]

# remove ddi
initial_row_count = ddi_df.shape[0]
ddi_filtered_df = ddi_df[~ddi_df['drug1'].isin(compounds_to_remove) & ~ddi_df['drug2'].isin(compounds_to_remove)]
removed_ddi_row_count = initial_row_count - ddi_filtered_df.shape[0]

# remove smiles
initial_row_count = smiles_df.shape[0]
filtered_smiiles_df = smiles_df[~smiles_df['compound'].isin(compounds_to_remove)]
removed_smiles_count = initial_row_count - filtered_smiiles_df.shape[0]


ddi_filtered_df.to_csv(ddi_output_file, sep='\t', index=False, header=False)
filtered_smiiles_df.to_csv(smiles_output_file, sep='\t', index=False, header=False)

print(f'Removed ddi rows: {removed_ddi_row_count}')
print(f'Removed smiles: {removed_smiles_count}')


base_edge_path = '/path/to/your/data/CGPD/subG_modify/edges/compound{}_edges.tsv'
base_node_path = '/path/to/your/data/CGPD/subG_modify/nodes/compound{}_nodes.tsv'
delete_files(compounds_to_remove, base_edge_path, base_node_path)

