# utils/model.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

# ---- Positional encoding used only in GAT path ----
class CustomPositionalEncoding(nn.Module):
    """Decayed sinusoidal positional encoding."""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        decay = (1 + torch.exp(-position / 100.0))
        pe[:, 0::2] = torch.sin(position * div_term) * decay
        pe[:, 1::2] = torch.cos(position * div_term) * decay
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, L, d]

    def forward(self, x):  # x: [B, L, d]
        return x + self.pe[:, :x.size(1), :]

# ---- Transformer encoder for SMILES (no PE here) ----
class SMILESEncoder(nn.Module):
    """Embedding -> Linear -> TransformerEncoder (no PE)."""
    def __init__(self, vocab_size, emb_dim, d_model, nhead, layers, dim_ff):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.fc = nn.Linear(emb_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, layers)

    def forward(self, x):  # [B, L]
        x = self.embedding(x)
        x = self.fc(x)
        return self.encoder(x)

# ---- GAT ----
class GraphAttentionLayer(nn.Module):
    """Single-head GAT with PE and self-loops; mean graph readout."""
    def __init__(self, in_f, out_f, dropout, alpha, concat=True):
        super().__init__()
        self.pos_enc = CustomPositionalEncoding(in_f)
        self.W = nn.Parameter(torch.zeros(in_f, out_f))
        nn.init.xavier_uniform_(self.W)
        self.a = nn.Parameter(torch.zeros(2*out_f, 1))
        nn.init.xavier_uniform_(self.a)
        self.leaky = nn.LeakyReLU(alpha)
        self.dropout = dropout
        self.concat = concat

    def forward(self, h, adj):  # h: [B, N, in_f], adj: [B, N, N]
        h = self.pos_enc(h)
        B, N, _ = h.size()
        eye = torch.eye(N, device=h.device).unsqueeze(0)
        adj = torch.where(eye > 0, torch.ones_like(adj), adj)

        Wh = h.reshape(-1, h.size(2)) @ self.W
        Wh = Wh.view(B, N, -1)

        a1 = self.a[:Wh.size(2)]
        a2 = self.a[Wh.size(2):]
        Wh1 = Wh @ a1
        Wh2 = Wh @ a2
        e = self.leaky(Wh1 + Wh2.permute(0,2,1))

        mask_val = -9e15 * torch.ones_like(e)
        attn = torch.where(adj > 0, e, mask_val)
        attn = torch.softmax(attn, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        h_prime = attn @ Wh

        deg = adj.sum(-1)
        valid = (deg > 0)
        valid[:, 0] = False
        denom = valid.sum(1, keepdim=True).clamp(min=1)
        g = (h_prime * valid.unsqueeze(-1)).sum(1) / denom
        return F.elu(g)

class MultiHeadGraphAttentionLayer(nn.Module):
    """Concatenate head outputs and project."""
    def __init__(self, in_f, out_f, heads, dropout, alpha, concat=True):
        super().__init__()
        assert out_f % heads == 0, "out_f must be divisible by heads"
        head_out = out_f // heads
        self.heads = nn.ModuleList([
            GraphAttentionLayer(in_f, head_out, dropout, alpha, concat)
            for _ in range(heads)
        ])
        self.fc = nn.Linear(out_f, out_f)

    def forward(self, h, adj):
        x = torch.cat([head(h, adj) for head in self.heads], dim=1)
        return self.fc(x)

# ---- Full LightningModule ----
class DDIClassifier(pl.LightningModule):
    """
    Fuse branches:
    - e1, e2: SMILES Transformer CLS embeddings
    - g1, g2: GAT graph embeddings over adjacency
    - v1, v2: precomputed compound vectors
    """
    def __init__(self, vocab, emb_dim, d_model, graph_dim, hid, num_cls,
                 clf_drop, gat_drop, lr, nhead, num_layers, ff, gat_alpha=0.3086):
        super().__init__()
        self.save_hyperparameters(ignore=['graph_dim', 'num_cls'])
        self.smiles_enc = SMILESEncoder(vocab, emb_dim, d_model, nhead, num_layers, ff)
        self.gat = MultiHeadGraphAttentionLayer(d_model, d_model, nhead, gat_drop, gat_alpha, True)

        combined = d_model*2 + d_model*2 + graph_dim*2
        self.fc1 = nn.Linear(combined, hid)
        self.bn1 = nn.BatchNorm1d(hid)
        self.fc2 = nn.Linear(hid, hid)
        self.bn2 = nn.BatchNorm1d(hid)
        self.fc3 = nn.Linear(hid, num_cls)

        self.act = nn.LeakyReLU()
        self.drop = nn.Dropout(clf_drop)
        self.loss_fn = nn.CrossEntropyLoss()
        self.lr = lr

    def forward(self, t1, t2, v1, v2, a1, a2):
        s1 = self.smiles_enc(t1); s2 = self.smiles_enc(t2)
        e1 = s1[:,0,:]; e2 = s2[:,0,:]
        g1 = self.gat(s1, a1);  g2 = self.gat(s2, a2)
        x = torch.cat([e1, e2, g1, g2, v1, v2], dim=1)
        x = self.act(self.bn1(self.fc1(x))); x = self.drop(x)
        x = self.act(self.bn2(self.fc2(x))); x = self.drop(x)
        return self.fc3(x)

    def training_step(self, batch, batch_idx):
        t1,t2,v1,v2,y,a1,a2 = batch
        logits = self(t1,t2,v1,v2,a1,a2)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss, prog_bar=False, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        t1,t2,v1,v2,y,a1,a2 = batch
        logits = self(t1,t2,v1,v2,a1,a2)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log("val_loss", loss, prog_bar=False, on_epoch=True, on_step=False)
        self.log("val_acc", acc, prog_bar=True, on_epoch=True, on_step=False)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)