# Preprocessing Pipelines for Knowledge Graph Embedding

This directory contains scripts for **preprocessing and embedding biomedical knowledge graphs** in preparation for downstream tasks such as **drug–drug interaction (DDI) prediction** or **pathway-centric drug representation learning**.  
The workflow extends the prior subgraph extraction pipeline (`randomwalk_mp.py`, `extract_hop.py`) by cleaning compound-level data and generating low-dimensional embeddings from compound-centered subgraphs.

---

## Overview

| Script File | Description |
|--------------|-------------|
| `subG_add_info.py` | Augments each compound subgraph with original entity and relation identifiers from the DRKG embedding index. Adds cross-references (`ori_node_num`) to ensure compatibility with embedding lookups. |
| `remove_compounds.py` | Removes specified problematic compounds and all related subgraph files (nodes, edges) from the dataset, ensuring graph consistency before embedding. |
| `subG_info_Extract_weight.py` | Computes compound-level vectors using entity and relation embeddings. Supports weighting as well as PCA or convolution-based transformations, with **`conv` used as the default method in this study**. Supported aggregation methods: `sum`, `mean`, `product`, `pca`, `conv`. |

---

## Data Preparation

Ensure that the following files are available before execution:

| File | Path | Description |
|------|------|-------------|
| **DDI file** | `/path/to/data/DDI/ddi.tsv` | DDI triples (drug1, drug2, interaction_type) |
| **Smiles file** | `/path/to/data/DDI/smiles.tsv` | Compound SMILES strings |
| **DRKG entities** | `/path/to/data/drkg/entities.tsv` | Mapping of node names to embedding indices |
| **DRKG relations** | `/path/to/data/drkg/relations.tsv` | Mapping of relation types to embedding indices |
| **DRKG embeddings** | `/path/to/data/drkg/embed/DRKG_TransE_l2_entity.npy`<br>`/path/to/data/drkg/embed/DRKG_TransE_l2_relation.npy` | Pretrained entity/relation embeddings from DRKG |
| **Subgraph files** | `/path/to/data/CGPD/hop4/steps_20000/prob_0.3/{nodes,edges}/compound{i}_*.tsv` | Compound-centered subgraphs generated via RWR |

---

## Step 1. Add Original Entity and Relation IDs

**Script:** `subG_add_info.py`

This step augments each subgraph (nodes, edges) by mapping local node IDs to their corresponding DRKG embedding indices.

**Key Actions**
- Reads each compound’s `nodes.tsv` and `edges.tsv`.
- Maps node names to their original numeric IDs (`ori_node_num`).
- Updates edge files with corresponding `ori_node_1`, `ori_node_2` fields.
- Writes modified files to `/CGPD/subG_modify/{nodes,edges}/`.

**Command**
```bash
python subG_add_info.py
```

**Output**
- `/path/to/data/CGPD/subG_modify/nodes/compound{i}_nodes.tsv`  
- `/path/to/data/CGPD/subG_modify/edges/compound{i}_edges.tsv`

---

## Step 2. Remove Invalid Compounds

**Script:** `remove_compounds.py`

Removes specific compounds and their associated node/edge files from the dataset, ensuring only valid compound graphs remain.

**Key Actions**
- Reads DDI and SMILES files.  
- Filters out rows containing invalid compound IDs (e.g., 651, 1170, 1359, ...).  
- Deletes corresponding subgraph files from `/subG_modify/nodes` and `/subG_modify/edges/`.

**Command**
```bash
python remove_compounds.py
```

**Output**
- Filtered DDI and SMILES TSVs  
  `/path/to/data/DDI/ddi_01.tsv`  
  `/path/to/data/DDI/smiles_01.tsv`  
- Deleted files (logged to stdout)

---

## Step 3. Extract Weighted Compound Embeddings

**Script:** `subG_info_Extract_weight.py`

Generates compound-level vector representations using various aggregation or convolutional methods.

**Supported Methods**

| Method | Description |
|--------|--------------|
| `pca` | Applies PCA to combined entity + relation embeddings |
| `sum` / `mean` / `product` | Aggregates embeddings and reduces dimensionality via Conv1D |
| `conv` | Applies 2D convolution (subCon model) for nonlinear feature extraction |

**Key Steps**
1. Loads DRKG entity/relation embeddings.  
2. Reads each compound’s modified subgraph files.  
3. Scales node attributes (`spread_value`, `visited_count`) using `MinMaxScaler`.  
4. Applies node-specific weights to embeddings.  
5. Combines and transforms embeddings according to the selected method.

**Command**
```bash
python subG_info_Extract_weight.py --vector_length 20 --method conv
```

**Output**
- Compound-level embedding CSV:  
  `/path/to/data/vec20_conv.csv`

---

## Directory Structure

```text
your_project_directory/
│
├── data/
│   ├── drkg/
│   │   ├── entities.tsv
│   │   ├── relations.tsv
│   │   ├── embed/
│   │   │   ├── DRKG_TransE_l2_entity.npy
│   │   │   └── DRKG_TransE_l2_relation.npy
│   │
│   ├── DDI/
│   │   ├── ddi.tsv
│   │   ├── smiles.tsv
│   │   ├── ddi_01.tsv
│   │   └── smiles_01.tsv
│   │
│   └── CGPD/
│       ├── hop4/
│       │   └── steps_20000/prob_0.3/{nodes,edges}/compound{i}_*.tsv
│       ├── subG_modify/
│       │   ├── nodes/compound{i}_nodes.tsv
│       │   └── edges/compound{i}_edges.tsv
│       └── vec20_conv.csv
```

---

## Notes

- All scripts assume consistent DRKG entity and relation indexing across files.  
- The compound count (1,706) can be adjusted in the loop range if new nodes are added.  
- Recommended to use a GPU-enabled environment for the convolutional embedding step (`--method conv`).  
- After vector extraction, embeddings can be directly used as input features for TRACE-DDI or related models.