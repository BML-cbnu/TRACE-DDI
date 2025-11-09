import os
import pandas as pd
from tqdm import tqdm
import argparse


# Step 1: Subgraph Modification
def modify_subgraphs(embed_path, subG_path, output_node_path, output_edge_path):
    """
    Modify subgraph files by mapping to original entity numbers.
    """
    os.makedirs(output_node_path, exist_ok=True)
    os.makedirs(output_edge_path, exist_ok=True)

    # Load entity reference
    ori_entity = pd.read_csv(os.path.join(embed_path, "entities.tsv"), sep='\t', header=None, names=['node', 'node_num'])
    ori_entity_dict = dict(zip(ori_entity['node'], ori_entity['node_num']))

    for i in tqdm(range(1705 + 1), desc="Modifying subgraphs"):
        entity_file = os.path.join(subG_path, f"nodes/compound{i}_nodes.tsv")
        relation_file = os.path.join(subG_path, f"edges/compound{i}_edges.tsv")

        if os.path.exists(entity_file) and os.path.exists(relation_file):
            sub_entity = pd.read_csv(entity_file, sep='\t', header=None,
                                     names=['node_id', 'node_name', 'node_type', 'visited_count', 'spread_value'])
            sub_relation = pd.read_csv(relation_file, sep='\t', header=None,
                                       names=['node1', 'relation_id', 'node2'])

            # Mapping to original entity numbers
            sub_entity['ori_node_num'] = sub_entity['node_name'].map(ori_entity_dict)
            sub_entity_reverse_dict = dict(zip(sub_entity['node_id'], sub_entity['ori_node_num']))
            sub_relation['ori_node_1'] = sub_relation['node1'].map(sub_entity_reverse_dict)
            sub_relation['ori_node_2'] = sub_relation['node2'].map(sub_entity_reverse_dict)

            # Add node names
            ori_node_num_to_name_dict = dict(zip(sub_entity['ori_node_num'], sub_entity['node_name']))
            sub_relation['ori_node_name_1'] = sub_relation['ori_node_1'].map(ori_node_num_to_name_dict)
            sub_relation['ori_node_name_2'] = sub_relation['ori_node_2'].map(ori_node_num_to_name_dict)

            # Save
            sub_entity.to_csv(os.path.join(output_node_path, f"compound{i}_nodes.tsv"),
                              sep='\t', index=False, header=False)
            sub_relation.to_csv(os.path.join(output_edge_path, f"compound{i}_edges.tsv"),
                                sep='\t', index=False, header=False)
        else:
            print(f"[Skip] compound{i}: missing files")

    print("Step 1 complete: Subgraph modification finished.")


## Step 2: Cleanup and Filtering
def delete_files(numbers, base_edge_path, base_node_path):
    """
    Delete node/edge files corresponding to specified compound numbers.
    """
    for number in numbers:
        edge_file = base_edge_path.format(number)
        node_file = base_node_path.format(number)
        for fpath in [edge_file, node_file]:
            if os.path.exists(fpath):
                os.remove(fpath)
                print(f"Deleted: {fpath}")
            else:
                print(f"File not found: {fpath}")


def cleanup_ddi_and_subgraphs(ddi_file, smiles_file, compounds_to_remove,
                              ddi_output_file, smiles_output_file,
                              base_edge_path, base_node_path):
    """
    Remove specific compounds from DDI/SMILES files and delete subgraph files.
    """
    ddi_df = pd.read_csv(ddi_file, sep='\t', names=['drug1', 'drug2', 'interaction_type'])
    smiles_df = pd.read_csv(smiles_file, sep='\t', header=None, names=['compound', 'smiles'])

    # Filter rows
    ddi_filtered = ddi_df[~ddi_df['drug1'].isin(compounds_to_remove) & ~ddi_df['drug2'].isin(compounds_to_remove)]
    smiles_filtered = smiles_df[~smiles_df['compound'].isin(compounds_to_remove)]

    # Save filtered data
    ddi_filtered.to_csv(ddi_output_file, sep='\t', index=False, header=False)
    smiles_filtered.to_csv(smiles_output_file, sep='\t', index=False, header=False)

    print(f"Removed {ddi_df.shape[0] - ddi_filtered.shape[0]} DDI rows")
    print(f"Removed {smiles_df.shape[0] - smiles_filtered.shape[0]} SMILES rows")

    # Delete subgraph files
    delete_files(compounds_to_remove, base_edge_path, base_node_path)
    print("Step 2 complete: Cleanup finished.")



parser = argparse.ArgumentParser(description="Subgraph preprocessing and cleanup pipeline")
parser.add_argument("--embed_path", type=str, default="./data/drkg/embed", 
                    help="Path to DRKG embedding directory")
parser.add_argument("--subG_path",type=str,default="./data/CGPD/hop4/steps_20000/prob_0.3",
                    help="Path to raw subgraph directory")
parser.add_argument("--output_node_path", type=str, default="./data/CGPD/subG_modify/nodes",
                    help="Path to save modified node files")
parser.add_argument("--output_edge_path", type=str, default="./data/CGPD/subG_modify/edges",
                    help="Path to save modified edge files")
parser.add_argument("--ddi_file", type=str, default="./data/DDI/ddi.tsv",
                    help="Path to original DDI file")
parser.add_argument("--smiles_file", type=str, default="./data/DDI/smiles.tsv", 
                    help="Path to original SMILES file")
parser.add_argument("--ddi_output_file", type=str, default="./data/DDI/ddi_01.tsv", 
                    help="Path to save filtered DDI file")
parser.add_argument("--smiles_output_file", type=str, default="./data/DDI/smiles_01.tsv", 
                    help="Path to save filtered SMILES file")
parser.add_argument("--base_edge_path", type=str, default="./data/CGPD/subG_modify/edges/compound{}_edges.tsv",
                    help="Template path for modified edge files (use {} for compound ID)")
parser.add_argument("--base_node_path",type=str, default="./data/CGPD/subG_modify/nodes/compound{}_nodes.tsv", 
                    help="Template path for modified node files (use {} for compound ID)")
args = parser.parse_args()
for arg, value in vars(args).items():
    print(f"{arg}: {value}")
    
# Fixed list of compounds to remove
compounds_to_remove = [651, 1170, 1359, 1475, 1509, 1603, 1612, 1623, 1627, 1634, 1653, 1684, 1696, 1698]

print("\nRunning preprocessing pipeline...")
# Step 1
modify_subgraphs(
    embed_path=args.embed_path,
    subG_path=args.subG_path,
    output_node_path=args.output_node_path,
    output_edge_path=args.output_edge_path,
)

# Step 2
cleanup_ddi_and_subgraphs(
    ddi_file=args.ddi_file,
    smiles_file=args.smiles_file,
    compounds_to_remove=compounds_to_remove,
    ddi_output_file=args.ddi_output_file,
    smiles_output_file=args.smiles_output_file,
    base_edge_path=args.base_edge_path,
    base_node_path=args.base_node_path
)
print("All processing complete.")
