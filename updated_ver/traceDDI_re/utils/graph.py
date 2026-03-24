from __future__ import annotations

import math
import re
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomPositionalEncoding(nn.Module):
    # Decayed sinusoidal PE
    pe: torch.Tensor

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        decay = 1.0 + torch.exp(-position / 100.0)

        pe[:, 0::2] = torch.sin(position * div_term) * decay
        pe[:, 1::2] = torch.cos(position * div_term) * decay
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class GraphAttentionLayer(nn.Module):
    # Single-head GAT + mean pooling
    def __init__(self, in_f: int, out_f: int, dropout: float, alpha: float) -> None:
        super().__init__()
        self.pos_enc = CustomPositionalEncoding(in_f)
        self.W = nn.Parameter(torch.zeros(in_f, out_f))
        nn.init.xavier_uniform_(self.W)
        self.a = nn.Parameter(torch.zeros(2 * out_f, 1))
        nn.init.xavier_uniform_(self.a)
        self.leaky = nn.LeakyReLU(alpha)
        self.dropout = dropout

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = self.pos_enc(h)
        B, N, _ = h.size()

        eye = torch.eye(N, device=h.device).unsqueeze(0)
        adj = torch.where(eye > 0, torch.ones_like(adj), adj)

        Wh = (h.reshape(-1, h.size(2)) @ self.W).view(B, N, -1)
        out_f = Wh.size(2)

        a1 = self.a[:out_f]
        a2 = self.a[out_f:]
        e = self.leaky((Wh @ a1) + (Wh @ a2).permute(0, 2, 1))

        mask_val = -9e15 * torch.ones_like(e)
        attn = torch.where(adj > 0, e, mask_val)
        attn = torch.softmax(attn, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        h_prime = attn @ Wh

        deg = adj.sum(-1)
        valid = deg > 0
        valid[:, 0] = False
        denom = valid.sum(1, keepdim=True).clamp(min=1)
        g = (h_prime * valid.unsqueeze(-1)).sum(1) / denom
        return F.elu(g)


class MultiHeadGraphAttentionLayer(nn.Module):
    # Multi-head GAT
    def __init__(
        self,
        in_f: int,
        out_f: int,
        heads: int,
        dropout: float,
        alpha: float,
    ) -> None:
        super().__init__()
        if out_f % heads != 0:
            raise ValueError("out_f must be divisible by heads")

        head_out = out_f // heads
        self.heads = nn.ModuleList(
            [GraphAttentionLayer(in_f, head_out, dropout, alpha) for _ in range(heads)]
        )
        self.fc = nn.Linear(out_f, out_f)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x = torch.cat([head(h, adj) for head in self.heads], dim=1)
        return self.fc(x)


_ADJ_PATTERN: re.Pattern[str] = re.compile(
    r"(%\d{2}|\[[^\[\]]*\]|Br|Cl|Si|Al|Na|K|Ca|Mg|Cu|Co|Zn|Fe|Mn|P|\.|=|#|-|\+|\(|\)|\[|\]|\{|\}|[A-Za-z]|\d+|@|\\|/)",
    re.I,
)


def smiles_to_adj_matrix(smiles: str, max_len: int) -> torch.Tensor:
    # Simplified SMILES-to-adjacency parser
    adj = torch.zeros((max_len, max_len), dtype=torch.float)
    bonds: Dict[int, int] = {}
    branches: List[Tuple[int, float]] = []
    chiral: List[str] = []

    cur_idx = 1
    last_bond: float = 1.0

    tokens = _ADJ_PATTERN.findall(smiles)
    for t in tokens:
        if re.match(r"[A-Za-z]|\[.*\]", t):
            if cur_idx >= max_len:
                break
            if cur_idx > 1:
                adj[cur_idx - 1, cur_idx] = last_bond
                adj[cur_idx, cur_idx - 1] = last_bond
                if chiral:
                    adj[cur_idx - 1, cur_idx] += 0.5
                    adj[cur_idx, cur_idx - 1] += 0.5
                    chiral.pop()
            cur_idx += 1
            last_bond = 1.0

        elif t.isdigit():
            rn = int(t)
            if rn in bonds:
                s = bonds.pop(rn)
                if 0 <= s < max_len and 0 <= (cur_idx - 1) < max_len:
                    adj[s, cur_idx - 1] = last_bond
                    adj[cur_idx - 1, s] = last_bond
            else:
                bonds[rn] = cur_idx - 1

        elif t.startswith("%") and t[1:].isdigit():
            rn = int(t[1:])
            if rn in bonds:
                s = bonds.pop(rn)
                if 0 <= s < max_len and 0 <= (cur_idx - 1) < max_len:
                    adj[s, cur_idx - 1] = last_bond
                    adj[cur_idx - 1, s] = last_bond
            else:
                bonds[rn] = cur_idx - 1

        elif t in "-=#:+:":
            last_bond = {"-": 1.0, "=": 2.0, "#": 3.0, ":": 1.5, "+": 1.0}[t]

        elif t == "(" and cur_idx > 1:
            branches.append((cur_idx - 1, last_bond))

        elif t == ")" and branches:
            _, lb = branches.pop()
            last_bond = lb

        elif t in ("@", "\\", "/"):
            chiral.append(t)

    return adj
