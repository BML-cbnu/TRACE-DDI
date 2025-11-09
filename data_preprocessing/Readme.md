---
output:
  html_document: default
  pdf_document: default
---
# Preprocessing Pipelines for Knowledge Graph

This repository contains two core scripts designed to preprocess a biomedical knowledge graph for applications in drug discovery and drug–drug interaction (DDI) prediction.
The preprocessing pipeline integrates Random Walk with Restart (RWR) to generate compound-centered subgraphs and identifies the shortest paths connecting compounds to biological pathways.


## Files

| Script File | Description |
|-------------|--------------|
| `randomwalk_mp.py` | Scripts to generate a subgraph for each compound using RWR on a knowledge graph. Supports multiprocessing to efficiently process large-scale data.|
|  `extract_hop.py` | Script to extract the shortest paths (within k-hops) from each compound to its related biological pathways within the generated subgraphs. |


## Data Preparation
Ensure the follwing files are available:
- DRKG entitey TSV file:: `/path/to/data/drkg/nodes.tsv` 
- DRKG relations TSV file: `/path/to/data/drkg/edges.tsv`

### 🔗 Downloading the DRKG Dataset

The **Drug Repurposing Knowledge Graph (DRKG)** dataset can be downloaded from the official source:

> **DRKG (Drug Repurposing Knowledge Graph)**  
> Repository: [https://github.com/gnn4dr/DRKG](https://github.com/gnn4dr/DRKG)  
>  
After extraction, the structure should look like this:

```text
drkg/
├── drkg.tsv
├── entity2src.tsv
├── relation_glossary.tsv
└── embed/
    ├── DRKG_TransE_l2_entity.npy
    ├── DRKG_TransE_l2_relation.npy
    ├── entities.tsv
    ├── relations.tsv
    ├── mol_contextpred.npy
    ├── mol_masking.npy
    ├── mol_infomax.npy
    ├── mol_edgepred.npy
    └── Readme.md

> Move the files to your project path, e.g., `/path/to/data/drkg/`.


## Usage
### Step 1: Extract subgraph on Knowledge Graph

Run `randomwalk_mp.py` to generate a subgraph for each compound using Random Walk with Restart (RWR).


### Key functions
The key functions and their roles in randomwalk_mp.py are as follows:

**1. extract_nodes_and_edges:**
- Filters the full knowledge graph to retain only the selected biological node types `(e.g., compounds, genes, pathways)`.

-  Saves the filtered data to `nodes_{nodes_name}.tsv` and `edges_{nodes_name}.tsv` under `/path/to/save/data/{nodes_name}` directory.

- ex. If you are select compounds, gene, disease and pathway as node types, set the nodes_name as argument nodes_name is `CGPDS`.

**2. load_or_create_subgraph:**
- Loads and builds a subgraph with previously selected node types. 

**3. random_walk_process**
- Perform Random Walk with Restart on the subgraph. Starting from 1,705 compound seed nodes, it generates subgraph for each compound by iteratively exploring neighboring nodes with restart probability using multiprocessing.

**Output Summary:**
- `Total compouns not reaching any pathway list`: Number of compounds for which the RWR failed to reach any pathway nodes.

**Output Files:**
- node files in sub graph: `/path/to/save/data/{nodes_name}/rw_mean/steps_{steps}/prob_{0.3}/node/compound{}.tsv`

- edges files in sub graph: `/path/to/save/data/{nodes_name}/rw_mean/steps_{steps}/prob_{prob}/edges/compound{}.tsv`


**Command:**
```bash
python randomwalk_mp.py --nodes_name CGPD --prob 0.3 --steps 20000 --num_workers 50 --iteration 10000
```

<br><br>  


## Step 2: Extract k-hop shortest paths

Run `extract_hop.py` to extract the shortest paths between compounds and pathway nodes from each subgraph.

### Key actions
The key steps are as follows:

- For each compound, identifies all accessible pathway nodes within its subgraph.

- Computes the distance btween a compound and all identified pathways using networkX library.

- Filters paths based on the maximum allowed  k-hop.

- Save extracted nodes and edges corresponding to valid shortest paths files.

**Output Summary:**
- `Total compounds processed`: Total number of compounds in DDI dataset.
-  `Missing pathway info`: Number of compounds without pathway information.
- `Missing meta-path`: 	Number of compounds without valid meta-paths (within the specified hop limit).

- `Missing meta-path compounds`: Lists of compound IDs missing pathway or meta-path data


**Output Files:**
- node files in shortest path graph: `/path/to/save/data/{nodes_type}/rw_mean/steps_{steps}/prob_{prob}/hop{num_hop}/node/compound{}.tsv`

- edges files in shortest path graph: `/path/to/save/data/{nodes_type}/rw_mean/steps_20000/prob_0.3/hop5/edges/compound{}.tsv`

**Command:**
```bash
python extract_hop.py --nodes_type CGPD --steps 20000 --num_hop 5 --prob 0.3
```




##  Directory Structure
```text
your project derectory/
│
├── data/                        
│   ├── drkg/                     
│   │   ├── nodes.tsv
│   │   └── edges.tsv
│   │
│   └── CGPD/                     
│       ├── nodes_CGPD.tsv
│       ├── edges_CGPD.tsv
│       └── rw_mean/              
│           ├── steps_20000/
│           │   ├── prob_0.3/
│           │   │   ├── nodes/
│           │   │   │   ├── compound0_nodes.tsv
│           │   │   │   └── ...
│           │   │   └── edges/
│           │   │       ├── compound0_edges.tsv
│           │   │       └── ...
│           │   │
│           │   └── hop5/          
│           │       ├── nodes/
│           │       │   ├── compound0_nodes.tsv
│           │       │   └── ...
│           │       └── edges/
│           │           ├── compound0_edges.tsv
│           │           └── ...
│           └── ...                
│
├── data_preprocessing/                      
   ├── randomwalk_mp.py          
   └── extract_hop.py            

```







