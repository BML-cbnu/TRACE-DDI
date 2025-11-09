import os
import pandas as pd

# Set the range for the compound files
start = 0
end = 1705 #1705

# Define paths

base_path = "/path/to/your/data"
embed_path = os.path.join(base_path, "drkg/embed")
subG_path = os.path.join(base_path, "CGPD/hop4/steps_20000/prob_0.3")
output_node_path = "/path/to/your/data/CGPD/subG_modify/nodes"
output_edge_path = "/path/to/your/data/CGPD/subG_modify/edges"

# Create directories if they do not exist
os.makedirs(output_node_path, exist_ok=True)
os.makedirs(output_edge_path, exist_ok=True)

# Load original entity and relation data
ori_entity = pd.read_csv(os.path.join(embed_path, "entities.tsv"), sep='\t', header=None, names=['node', 'node_num'])
ori_relation = pd.read_csv(os.path.join(embed_path, "relations.tsv"))

# Create a dictionary from ori_entity for quick lookup
ori_entity_dict = dict(zip(ori_entity['node'], ori_entity['node_num']))

# Process files from compound0 to compound1705
for i in range(start, end + 1):
    # Define file paths
    entity_file = os.path.join(subG_path, f"nodes/compound{i}_nodes.tsv")
    relation_file = os.path.join(subG_path, f"edges/compound{i}_edges.tsv")
    # Check if both files exist
    if os.path.exists(entity_file) and os.path.exists(relation_file):
        # Load sub entity and relation data
        sub_entity = pd.read_csv(entity_file, sep='\t', header=None, names=['node_id', 'node_name', 'node_type', 'visited_count', 'spread_value'])
        sub_relation = pd.read_csv(relation_file, sep='\t', header=None, names=['node1', 'relation_id', 'node2'])
        
        # Add a new column "ori_node_num" to sub_entity and fill it with corresponding values from ori_entity
        sub_entity['ori_node_num'] = sub_entity['node_name'].map(ori_entity_dict)
        
        # Create a reverse dictionary from sub_entity for lookup of node_id to ori_node_num
        sub_entity_reverse_dict = dict(zip(sub_entity['node_id'], sub_entity['ori_node_num']))
        
        # Add new columns "ori_node_1" and "ori_node_2" to sub_relation by mapping node1 and node2 to their corresponding ori_node_num values
        sub_relation['ori_node_1'] = sub_relation['node1'].map(sub_entity_reverse_dict)
        sub_relation['ori_node_2'] = sub_relation['node2'].map(sub_entity_reverse_dict)
        
        # Create a reverse dictionary for lookup of ori_node_num to node_name
        ori_node_num_to_name_dict = dict(zip(sub_entity['ori_node_num'], sub_entity['node_name']))
        
        # Add new columns "ori_node_name_1" and "ori_node_name_2" to sub_relation by mapping ori_node_1 and ori_node_2 to their corresponding node_name values
        sub_relation['ori_node_name_1'] = sub_relation['ori_node_1'].map(ori_node_num_to_name_dict)
        sub_relation['ori_node_name_2'] = sub_relation['ori_node_2'].map(ori_node_num_to_name_dict)
        
        # Save the modified sub_entity and sub_relation to the output directories
        sub_entity.to_csv(os.path.join(output_node_path, f"compound{i}_nodes.tsv"), sep='\t', index=False, header=False)
        sub_relation.to_csv(os.path.join(output_edge_path, f"compound{i}_edges.tsv"), sep='\t', index=False, header=False)
        
    else:
        print(f"Files for compound{i} do not exist, skipping...")

print("Processing complete.")
