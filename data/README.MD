# Data Files for TRACE-DDI

This directory contains the core input data used in the **TRACE-DDI** framework for **drug–drug interaction (DDI) prediction**.  
All files are tab-separated (`.tsv`) and must share consistent compound identifiers.

---

## File Descriptions

### 1. `ddi.tsv`
**Purpose:**  
Defines all known or experimentally verified **drug–drug interactions** used for model training and validation.

**Columns:**
| Column | Description |
|---------|-------------|
| `drug1` | Unique compound identifier for the first drug in the pair |
| `drug2` | Unique compound identifier for the second drug in the pair |
| `interaction` | Integer label indicating the interaction class |

**Notes:**
- No header row is present in the original file; the three columns correspond to `(drug1, drug2, interaction)`.  
- Interaction labels can be binary (0/1) or categorical depending on the dataset configuration.  
- Used directly as the main input for model training in `trace-ddi.py`.

---

### 2. `smiles.tsv`
**Purpose:**  
Contains the **canonical SMILES strings** corresponding to each compound in the DDI dataset.

**Columns:**
| Column | Description |
|---------|-------------|
| `compound_id` | Unique compound identifier (matches `drug1`/`drug2` in `ddi.tsv`) |
| `smiles` | Canonical SMILES representation of the molecular structure |

**Notes:**
- Original file has no header row.  
- SMILES strings are used by the Transformer encoder to extract chemical features.  
- Ensure all compound IDs correspond exactly to those in `ddi.tsv`.

---

### 3. `filtered-hetionet-v1.0-nodes.tsv`
**Purpose:**  
Provides **human-readable biological entity information** (from Hetionet v1.0) used in interpretability and visualization analyses.

**Columns:**
| Column | Description |
|---------|-------------|
| `id` | Node identifier (Hetionet or DRKG-mapped) |
| `name` | Human-readable name of the biological entity |
| `kind` | Node type (e.g., Compound, Gene, Pathway, Disease) |

**Notes:**
- Used by `analysis/CS_analysis.py` to map DRKG nodes to descriptive biological names.  
- Only filtered subset of Hetionet nodes are included.  

---

### 4. `vec20_conv.csv`
**Purpose:**  
Provides a **fixed-length (20-dimensional) continuous vector representation** for each compound, used as an **auxiliary compound-level feature** in the TRACE-DDI framework.

**Structure:**
- **Total columns:** 21  
- **Composition:**  
  - First column: compound index (`idx`, integer, used for internal referencing)  
  - Remaining columns: 20 real-valued features (`v1`–`v20`)

**Notes:**
- CSV format using comma (`,`) delimiters.  
- Each row corresponds to a single compound.  
- The first column (`idx`) serves as an **internal index aligned with the compound ordering** used in the dataset and does not carry intrinsic chemical meaning.  
- The 20-dimensional vectors represent precomputed numerical features, such as graph embeddings, network-derived representations, or other transformed compound-level features (`conv`).  
- These vectors can be **optionally integrated** with SMILES-based representations via feature concatenation or late-fusion strategies in TRACE-DDI.

---

### Directory Layout

```text
data/
├── ddi.tsv                        # Drug–drug interaction table (no header)
├── smiles.tsv                     # SMILES strings per compound (no header)
├── filtered-hetionet-v1.0-nodes.tsv  # Biological entity metadata
└── vec20_conv.csv                 # 20-dim compound vector features (CSV)

---

## Integration Notes

- All TSV files use UTF-8 encoding and tab (`\t`) delimiters.  
- Ensure consistent compound IDs across `ddi.tsv` and `smiles.tsv`.  
- Hetionet node names (`filtered-hetionet-v1.0-nodes.tsv`) are optional but recommended for interpretable visualizations.  
- These files serve as input for both:
  - **Model training**: `trace-ddi.py`
  - **Network analysis**: `analysis/CS_analysis.py`
