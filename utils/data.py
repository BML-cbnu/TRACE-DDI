# utils/data.py
import os
import re
import logging
import math
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# --------- Tokenizer ---------
class SMILESTokenizer:
    """Regex-based SMILES tokenizer with reserved special tokens."""
    def __init__(self, smiles_list, case_sensitive=True):
        self.special_tokens = {'<PAD>':0,'<UNK>':1,'<CLS>':2,'<SEP>':3}
        self.case_sensitive = case_sensitive
        self.vocab = self._build_vocab(smiles_list)

    @staticmethod
    def _compile_pattern(case_sensitive=True):
        flags = 0 if case_sensitive else re.I
        return re.compile(
            r'(%\d{2}|\[[^\[\]]*\]|Br|Cl|Si|Al|Na|K|Ca|Mg|Cu|Co|Zn|Fe|Mn|P|\.|=|#|-|\+|\(|\)|\[|\]|\{|\}|[A-Za-z]|\d+|@|\\|/)',
            flags
        )

    def _build_vocab(self, smiles_list):
        tokens = set()
        p = self._compile_pattern(self.case_sensitive)
        for s in smiles_list:
            tokens.update(p.findall(s))
        off = len(self.special_tokens)
        return {**self.special_tokens, **{tok:i+off for i,tok in enumerate(sorted(tokens))}}

    def tokenize(self, s):
        p = self._compile_pattern(self.case_sensitive)
        return [self.vocab.get(t, self.special_tokens['<UNK>']) for t in p.findall(s)]

    def encode(self, s, max_length=128):
        cls = self.special_tokens['<CLS>']
        t = [cls] + self.tokenize(s)
        if len(t) > max_length:
            t = t[:max_length]
        pad = self.special_tokens['<PAD>']
        return t + [pad]*(max_length-len(t))

# --------- SMILES → adjacency ---------
_ADJ_PATTERN = re.compile(
    r'(%\d{2}|\[[^\[\]]*\]|Br|Cl|Si|Al|Na|K|Ca|Mg|Cu|Co|Zn|Fe|Mn|P|\.|=|#|-|\+|\(|\)|\[|\]|\{|\}|[A-Za-z]|\d+|@|\\|/)',
    re.I
)

def smiles_to_adj_matrix(smiles, max_len):
    """Build a fixed-size adjacency matrix from a SMILES string."""
    import torch
    adj = torch.zeros((max_len, max_len), dtype=torch.float)
    bonds = {}
    branches = []
    chiral = []
    cur_idx = 1
    last_bond = 1

    tokens = _ADJ_PATTERN.findall(smiles)
    for t in tokens:
        if re.match(r'[A-Za-z]|\[.*\]', t):
            if cur_idx >= max_len:
                break
            if cur_idx > 1:
                adj[cur_idx-1, cur_idx] = last_bond
                adj[cur_idx, cur_idx-1] = last_bond
                if chiral:
                    adj[cur_idx-1, cur_idx] += 0.5
                    adj[cur_idx, cur_idx-1] += 0.5
                    chiral.pop()
            cur_idx += 1
            last_bond = 1
        elif t.isdigit():
            rn = int(t)
            if rn in bonds:
                s = bonds.pop(rn)
                if 0 <= s < max_len and 0 <= cur_idx-1 < max_len:
                    adj[s, cur_idx-1] = last_bond
                    adj[cur_idx-1, s] = last_bond
            else:
                bonds[rn] = cur_idx - 1
        elif t.startswith('%') and t[1:].isdigit():
            rn = int(t[1:])
            if rn in bonds:
                s = bonds.pop(rn)
                if 0 <= s < max_len and 0 <= cur_idx-1 < max_len:
                    adj[s, cur_idx-1] = last_bond
                    adj[cur_idx-1, s] = last_bond
            else:
                bonds[rn] = cur_idx - 1
        elif t in '-=#:+':
            last_bond = {'-':1,'=':2,'#':3,':':1.5}[t]
        elif t == '(' and cur_idx > 1:
            branches.append((cur_idx-1, last_bond))
        elif t == ')' and branches:
            _, lb = branches.pop()
            last_bond = lb
        elif t in ['@','\\','/']:
            chiral.append(t)
        else:
            pass
    return adj

def pad_adjs(adjs):
    """Pad a list of [N×N] adjacencies to the largest N in the batch."""
    import torch
    max_n = max(a.size(0) for a in adjs)
    return torch.stack([F.pad(a,(0,max_n-a.size(1),0,max_n-a.size(0))) for a in adjs])

# --------- Dataset ---------
class DDIDataset(Dataset):
    """Return tokens, vectors, adjs, and label for each pair."""
    def __init__(self, df, tokenizer, comp_vecs, d2i, max_len=128):
        self.df = df.reset_index(drop=True)
        self.tk = tokenizer
        self.cv = comp_vecs
        self.map = d2i
        self.max_len = max_len

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        import torch
        row = self.df.iloc[idx]
        t1 = torch.tensor(self.tk.encode(row['smiles_x'], self.max_len), dtype=torch.long)
        t2 = torch.tensor(self.tk.encode(row['smiles_y'], self.max_len), dtype=torch.long)
        v1 = torch.tensor(self.cv[self.map[row['drug1']]], dtype=torch.float)
        v2 = torch.tensor(self.cv[self.map[row['drug2']]], dtype=torch.float)
        a1 = smiles_to_adj_matrix(row['smiles_x'], self.max_len)
        a2 = smiles_to_adj_matrix(row['smiles_y'], self.max_len)
        y  = torch.tensor(row['interaction'], dtype=torch.long)
        return t1,t2,v1,v2,y,a1,a2

def custom_collate_fn(batch):
    """Pad sequences and adjacencies; stack vectors and labels."""
    import torch
    t1,t2,v1,v2,y,a1,a2 = zip(*batch)
    t1 = torch.nn.utils.rnn.pad_sequence(t1, batch_first=True, padding_value=0)
    t2 = torch.nn.utils.rnn.pad_sequence(t2, batch_first=True, padding_value=0)
    v1 = torch.stack(v1); v2 = torch.stack(v2)
    y  = torch.stack(y)
    a1 = pad_adjs(a1); a2 = pad_adjs(a2)
    return t1,t2,v1,v2,y,a1,a2

# --------- Loader w/ logs ---------
def load_and_prepare_data(ddi_path, smiles_path, compound_vec_path, case_sensitive=True):
    """Load files, merge, encode labels, load vectors, build tokenizer; log each step."""
    logger.info("Loading DDI and SMILES data...")
    ddi = pd.read_csv(ddi_path, sep='\t', header=None, names=['drug1','drug2','interaction'])
    smiles = pd.read_csv(smiles_path, sep='\t', header=None, names=['drug','smiles'])

    logger.info("Merging SMILES into DDI dataframe...")
    ddi = ddi.merge(smiles, left_on='drug1', right_on='drug') \
             .rename(columns={'smiles':'smiles_x'}).drop('drug', axis=1)
    ddi = ddi.merge(smiles, left_on='drug2', right_on='drug') \
             .rename(columns={'smiles':'smiles_y'}).drop('drug', axis=1)

    logger.info("Casting interaction to category and encoding labels...")
    ddi['interaction'] = ddi['interaction'].astype('category')
    code_to_label = {i:l for i,l in enumerate(ddi['interaction'].cat.categories)}
    ddi['interaction'] = ddi['interaction'].cat.codes

    logger.info("Loading compound vectors (precomputed KG embeddings)...")
    comp_df = pd.read_csv(compound_vec_path, index_col=0)
    comp_vecs = comp_df.to_numpy()
    drug_to_idx = {d:i for i,d in enumerate(comp_df.index)}

    logger.info("Initializing tokenizer...")
    all_smiles = pd.concat([ddi['smiles_x'], ddi['smiles_y']]).unique()
    tokenizer = SMILESTokenizer(all_smiles, case_sensitive=case_sensitive)
    logger.info(f"Tokenizer built: vocab_size={len(tokenizer.vocab)}")

    return ddi, tokenizer, code_to_label, comp_vecs, drug_to_idx