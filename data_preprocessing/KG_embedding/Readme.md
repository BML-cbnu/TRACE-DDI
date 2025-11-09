# Subgraph Embedding Generation Pipeline for DDI Prediction

This repository contains scripts to preprocess compound-centered subgraphs and generate embeddings for use in Drug-Drug Interaction (DDI) prediction models. The pipeline consists of modifying raw subgraphs, cleaning DDI/SMILES datasets, and computing 
compound-level vectors using various aggregation methods.

---

## Files

| Script File | Description |
|-------------|-------------|
| `subG_add_info.py` | Modify subgraph files by mapping nodes to original entity IDs. Cleans DDI/SMILES datasets and deletes subgraphs for specific compounds. |
| `subG_info_Extract_weight.py` | Generates compound-level vectors using modified subgraphs. Supports multiple aggregation methods (`pca`, `sum`, `mean`, `product`, `conv`). |

---

## Data Preparation

Ensure the following files are available:
- DDI dataset TSV: `/path/to/data/DDI/ddi.tsv`
- SMILES dataset TSV: `path/to/data/DDI/smiles.tsv`
- DRKG entity TSV: `/path/to/data/drkg/embed/entities.tsv`
- DRKG relation TSV: `/path/to/data/drkg/embed/relations.tsv`
- DRKG enetity TransE npy: `/path/to/data/drkg/embed/DRKG_TransE_l2_entity.npy`
- DRKG relation TransE npy: `/path/to/data/drkg/embed/DRKG_TransE_l2_relation.npy`
- Subgraph nodes files: `/path/to/data/CGPD/hop4/steps_20000/prob_0.3/nodes/comopound{}_nodes.tsv`
- Subgraph edges files: `/path/to/data/CGPD/hop4/steps_20000/prob_0.3/edges/compouond{}_edges.tsv`

---

## Step 1: Modify Subgraphs and Cleanup

`subG_add_info.py` maps subgraph nodes to original DRKG entity IDs, cleans DDI/SMILES datasets, and deletes invalid subgraphs.

### Key Functions

**1. `modify_subgraphs()`**  
- Maps subgraph nodes to original entity numbers  
- Adds original node names to edge files  
- Saves modified node/edge files under `output_node_path` and `output_edge_path`  

**2. `cleanup_ddi_and_subgraphs()`**  
- Filters DDI and SMILES datasets to remove **fixed list of compounds**  
- Deletes corresponding subgraph files  

**Fixed compounds to remove:**  
- The compounds(651, 1170, 1359, 1475, 1509, 1603, 1612, 1623, 1627, 1634, 1653, 1684, 1696, 1698) were removed because either:
- Random Walk failed to reach any pathway nodes in their subgraphs, or no valid shortest path to pathway nodes could be extracted during the k-hop path extraction step.

### Us

```bash
python subG_add_info.py \
  --embed_path /path/to/data/drkg/embed \
  --subG_path /path/to/data/CGPD/hop4/steps_20000/prob_0.3 \
  --output_node_path /path/to/data/CGPD/subG_modify/nodes \
  --output_edge_path /path/to/data/CGPD/subG_modify/edges \
  --ddi_file /path/to/data/DDI/ddi.tsv \
  --smiles_file /path/to/data/DDI/smiles.tsv \
  --ddi_output_file /path/to/data/DDI/ddi_01.tsv \
  --smiles_output_file /path/to/data/DDI/smiles_01.tsv \
  --base_edge_path /path/to/data/CGPD/subG_modify/edges/compound{}_edges.tsv \
  --base_node_path /path/to/data/CGPD/subG_modify/nodes/compound{}_nodes.tsv

```


---

## Step 2: Generate Compound Vectors

`subG_info_Extract_weight.py` generates fixed-length vectors for each compound from the modified subgraphs.

### Key modules

**1. `SubCon2d`**  
- 2D CNN-based subgraph encoder  
- Generates fixed-length vectors from combined node and edge embeddings

**2. `SubCon1d`**  
- 1D CNN for dimensionality reduction of aggregated embeddings (`sum`, `mean`, `product`)  

**3. `process_single_compound()`**  
- Loads subgraph nodes and edges  
- Applies node weights based on `spread_value` and `visited_count` for drug nodes  
- Generates a vector using the selected method: `pca`, `sum`, `mean`, `product`, `conv`  

**4. `generate_compound_vectors()`**  
- Processes all compounds sequentially  
- Saves the vectors to CSV

### Command

```bash
python subG_info_Extract_weight.py \
  --base_dir /path/to/data/CGPD/subG_modify \
  --embed_dir /path/to/data/drkg/embed \
  --save_dir /path/to/data/CGPD \
  --vector_length 20 \
  --method conv \
```


##  Directory Structure
```text
/path/to/
│
├── data/
│   ├── drkg/embed/
│   │   ├── entities.tsv
│   │   ├── relations.tsv
│   │   ├── DRKG_TransE_l2_entity.npy
│   │   └── DRKG_TransE_l2_relation.npy
│   │
│   ├── DDI/
│   │   ├── ddi.tsv
│   │   ├── ddi_01.tsv
│   │   ├── smiles.tsv
│   │   └── smiles_01.tsv
│   │
│   └── CGPD/
│       ├── hop4/steps_20000/prob_0.3/
│       │   ├── nodes/
│       │   │   ├── compound0_nodes.tsv
│       │   │   └── ...
│       │   └── edges/
│       │       ├── compound0_edges.tsv
│       │       └── ...
│       │
│       ├── subG_modify/
│       │   ├── nodes/
│       │   │   ├── compound0_nodes.tsv
│       │   │   └── ...
│       │   └── edges/
│       │       ├── compound0_edges.tsv
│       │       └── ...
│       │
│       └── vec_{vector_length}_{method}.csv
│
├── model/
│   ├── subG_add_info.py
│   ├── subG_info_Extract_weight.py
│   └── trace_ddi.py