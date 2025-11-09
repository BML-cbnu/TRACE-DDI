⸻


# Preprocessing Pipelines for Knowledge Graph

This repository contains two core scripts designed to preprocess a biomedical knowledge graph for applications in **drug discovery** and **drug–drug interaction (DDI) prediction**.  
The preprocessing pipeline integrates **Random Walk with Restart (RWR)** to generate compound-centered subgraphs and identifies the **shortest paths connecting compounds to biological pathways**.

---

## Files

| Script File | Description |
|-------------|-------------|
| `randomwalk_mp.py` | Generates a subgraph for each compound using RWR on a knowledge graph. Supports multiprocessing to efficiently process large-scale data. |
| `extract_hop.py` | Extracts the shortest paths (within k-hops) from each compound to its related biological pathways within the generated subgraphs. |

---

## Data Preparation

Ensure the following files are available before execution:

- DRKG entity TSV file: `/path/to/data/drkg/embed/entities.tsv`
- DRKG relations TSV file: `/path/to/data/drkg/embed/relations.tsv`
- DRKG embeddings: `/path/to/data/drkg/embed/DRKG_TransE_l2_entity.npy`, `/path/to/data/drkg/embed/DRKG_TransE_l2_relation.npy`

### 🔗 Downloading the DRKG Dataset

Download the **Drug Repurposing Knowledge Graph (DRKG)** dataset from the official repository:

> **Repository:** [https://github.com/gnn4dr/DRKG](https://github.com/gnn4dr/DRKG)  
> **Direct download (tar.gz):** [https://dgl-data.s3-us-west-2.amazonaws.com/dataset/DRKG/drkg.tar.gz](https://dgl-data.s3-us-west-2.amazonaws.com/dataset/DRKG/drkg.tar.gz)

After extraction:

```bash
tar -xzvf drkg.tar.gz

You will see the following structure:

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

Move the files to your project path, e.g.:

/path/to/data/drkg/


⸻

Usage

Step 1: Extract Subgraphs on Knowledge Graph

Run randomwalk_mp.py to generate a subgraph for each compound using Random Walk with Restart (RWR).

Key Functions
	1.	extract_nodes_and_edges
	•	Filters the full knowledge graph to retain only selected biological node types (e.g., compounds, genes, pathways).
	•	Saves the filtered data as nodes_{nodes_name}.tsv and edges_{nodes_name}.tsv under /path/to/save/data/{nodes_name}.
	•	Example: if you select compounds, genes, diseases, and pathways as node types, set --nodes_name CGPDS.
	2.	load_or_create_subgraph
	•	Loads or builds a subgraph containing only the selected node types.
	3.	random_walk_process
	•	Performs RWR on the subgraph.
	•	Starting from 1,705 compound seed nodes, it generates one subgraph per compound using multiprocessing with restart probability.

Output Summary
	•	Total compounds not reaching any pathway list: number of compounds for which RWR failed to reach any pathway nodes.

Output Files
	•	Node files:
/path/to/save/data/{nodes_name}/rw_mean/steps_{steps}/prob_{prob}/nodes/compound{i}_nodes.tsv
	•	Edge files:
/path/to/save/data/{nodes_name}/rw_mean/steps_{steps}/prob_{prob}/edges/compound{i}_edges.tsv

Command

python randomwalk_mp.py --nodes_name CGPD --prob 0.3 --steps 20000 --num_workers 50 --iteration 10000


⸻

Step 2: Extract k-hop Shortest Paths

Run extract_hop.py to extract the shortest paths between compounds and pathway nodes from each subgraph.

Key Actions
	•	For each compound, identifies all accessible pathway nodes within its subgraph.
	•	Computes distances between a compound and all identified pathways using the NetworkX library.
	•	Filters paths based on the maximum allowed k-hop.
	•	Saves nodes and edges corresponding to valid shortest paths.

Output Summary
	•	Total compounds processed: number of compounds processed.
	•	Missing pathway info: number of compounds without pathway information.
	•	Missing meta-path: number of compounds without valid meta-paths within the hop limit.
	•	Missing meta-path compounds: IDs of compounds missing pathway or meta-path data.

Output Files
	•	Node files:
/path/to/save/data/{nodes_type}/rw_mean/steps_{steps}/prob_{prob}/hop{num_hop}/nodes/compound{i}_nodes.tsv
	•	Edge files:
/path/to/save/data/{nodes_type}/rw_mean/steps_{steps}/prob_{prob}/hop{num_hop}/edges/compound{i}_edges.tsv

Command

python extract_hop.py --nodes_type CGPD --steps 20000 --num_hop 5 --prob 0.3


⸻

Directory Structure

your_project_directory/
│
├── data/
│   ├── drkg/
│   │   ├── drkg.tsv
│   │   ├── entity2src.tsv
│   │   ├── relation_glossary.tsv
│   │   └── embed/
│   │       ├── DRKG_TransE_l2_entity.npy
│   │       ├── DRKG_TransE_l2_relation.npy
│   │       ├── entities.tsv
│   │       └── relations.tsv
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
│           │   └── hop5/
│           │       ├── nodes/
│           │       │   ├── compound0_nodes.tsv
│           │       │   └── ...
│           │       └── edges/
│           │           ├── compound0_edges.tsv
│           │           └── ...
│           └── ...


⸻

Notes
	•	Ensure consistent entity and relation indices across all DRKG-derived files.
	•	The default number of compounds (1,705) corresponds to the DDI dataset used in TRACE-DDI.
	•	Parameters such as restart probability (--prob) and steps (--steps) can be adjusted according to the graph scale.
	•	Generated subgraphs and shortest-path files are required inputs for subsequent embedding and DDI prediction modules.

---