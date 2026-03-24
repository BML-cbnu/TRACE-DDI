from __future__ import annotations

from typing import Dict, Sequence, Tuple, cast

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from utils.graph import smiles_to_adj_matrix
from utils.tokenizer import SMILESTokenizer

Sample = Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]

Batch = Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]


def pad_adjs(adjs: Sequence[torch.Tensor]) -> torch.Tensor:
    max_n = max(a.size(0) for a in adjs)
    return torch.stack(
        [F.pad(a, (0, max_n - a.size(1), 0, max_n - a.size(0))) for a in adjs]
    )


def custom_collate_fn(batch: Sequence[Sample]) -> Batch:
    t1_list, t2_list, v1_list, v2_list, y_list, a1_list, a2_list = zip(*batch)

    t1 = torch.nn.utils.rnn.pad_sequence(
        list(t1_list), batch_first=True, padding_value=0
    )
    t2 = torch.nn.utils.rnn.pad_sequence(
        list(t2_list), batch_first=True, padding_value=0
    )
    v1 = torch.stack(list(v1_list))
    v2 = torch.stack(list(v2_list))
    y = torch.stack(list(y_list))
    a1 = pad_adjs(list(a1_list))
    a2 = pad_adjs(list(a2_list))
    return t1, t2, v1, v2, y, a1, a2


class DDIDataset(Dataset[Sample]):
    # (drug1, drug2) with SMILES, vectors, adjacency
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: SMILESTokenizer,
        comp_vecs: np.ndarray,
        d2i: Dict[str, int],
        max_len: int,
    ) -> None:
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.tk = tokenizer
        self.cv = comp_vecs
        self.map = d2i
        self.max_len = max_len

    def __len__(self) -> int:
        return int(len(self.df))

    def __getitem__(self, idx: int) -> Sample:
        row = self.df.iloc[idx]

        s1 = cast(str, row["smiles_x"])
        s2 = cast(str, row["smiles_y"])
        d1 = cast(str, row["drug1"])
        d2 = cast(str, row["drug2"])
        yv = int(row["interaction"])

        t1 = torch.tensor(self.tk.encode(s1, self.max_len), dtype=torch.long)
        t2 = torch.tensor(self.tk.encode(s2, self.max_len), dtype=torch.long)
        v1 = torch.tensor(self.cv[self.map[d1]], dtype=torch.float)
        v2 = torch.tensor(self.cv[self.map[d2]], dtype=torch.float)
        a1 = smiles_to_adj_matrix(s1, self.max_len)
        a2 = smiles_to_adj_matrix(s2, self.max_len)
        y = torch.tensor(yv, dtype=torch.long)

        return t1, t2, v1, v2, y, a1, a2
