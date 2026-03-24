from __future__ import annotations

from typing import cast

import pytorch_lightning as pl
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score

from utils.dataset import Batch
from utils.graph import MultiHeadGraphAttentionLayer


class SMILESEncoder(nn.Module):
    # Emb -> proj -> TransformerEncoder
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        d_model: int,
        nhead: int,
        layers: int,
        dim_ff: int,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.fc = nn.Linear(emb_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.fc(x)
        return self.encoder(x)


class DDIClassifier(pl.LightningModule):
    # Transformer CLS + GAT graph embedding + KG vectors
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        d_model: int,
        graph_dim: int,
        hid_dim: int,
        num_classes: int,
        clf_drop: float,
        gat_drop: float,
        lr: float,
        tf_nhead: int,
        tf_layers: int,
        tf_dim_ff: int,
        gat_heads: int,
        gat_alpha: float,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.smiles_enc = SMILESEncoder(
            vocab_size=vocab_size,
            emb_dim=emb_dim,
            d_model=d_model,
            nhead=tf_nhead,
            layers=tf_layers,
            dim_ff=tf_dim_ff,
        )
        self.gat = MultiHeadGraphAttentionLayer(
            in_f=d_model,
            out_f=d_model,
            heads=gat_heads,
            dropout=gat_drop,
            alpha=gat_alpha,
        )

        combined_dim = d_model * 4 + graph_dim * 2
        self.fc1 = nn.Linear(combined_dim, hid_dim)
        self.bn1 = nn.BatchNorm1d(hid_dim)
        self.fc2 = nn.Linear(hid_dim, hid_dim)
        self.bn2 = nn.BatchNorm1d(hid_dim)
        self.fc3 = nn.Linear(hid_dim, num_classes)

        self.act = nn.LeakyReLU()
        self.drop = nn.Dropout(clf_drop)
        self.loss_fn = nn.CrossEntropyLoss()
        self.lr = lr

    def forward(
        self,
        t1: torch.Tensor,
        t2: torch.Tensor,
        v1: torch.Tensor,
        v2: torch.Tensor,
        a1: torch.Tensor,
        a2: torch.Tensor,
    ) -> torch.Tensor:
        s1 = self.smiles_enc(t1)
        s2 = self.smiles_enc(t2)

        e1 = s1[:, 0, :]
        e2 = s2[:, 0, :]

        g1 = self.gat(s1, a1)
        g2 = self.gat(s2, a2)

        x = torch.cat([e1, e2, g1, g2, v1, v2], dim=1)

        x = self.act(self.bn1(self.fc1(x)))
        x = self.drop(x)
        x = self.act(self.bn2(self.fc2(x)))
        x = self.drop(x)
        return self.fc3(x)

    def training_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        t1, t2, v1, v2, y, a1, a2 = batch
        logits = self(t1, t2, v1, v2, a1, a2)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss, prog_bar=False, on_epoch=True, on_step=False)
        return cast(torch.Tensor, loss)

    def validation_step(self, batch: Batch, batch_idx: int) -> None:
        t1, t2, v1, v2, y, a1, a2 = batch
        logits = self(t1, t2, v1, v2, a1, a2)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy_score(y.detach().cpu().tolist(), preds.detach().cpu().tolist())

        self.log("val_loss", loss, prog_bar=False, on_epoch=True, on_step=False)
        self.log("val_acc", acc, prog_bar=True, on_epoch=True, on_step=False)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.lr)
