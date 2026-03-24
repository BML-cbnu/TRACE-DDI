import os
import re
from typing import Optional

import pandas as pd


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_fname(txt: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", txt)


def short_label(name: str, maxlen: int = 36) -> str:
    s = str(name)
    if len(s) <= maxlen:
        return s
    if s.startswith("Pathway::"):
        head = "Pathway::"
        tail = s.split("::", 1)[1]
        if len(tail) > maxlen - len(head):
            tail = tail[: maxlen - len(head) - 3] + "..."
        return head + tail
    return s[: maxlen - 3] + "..."


def read_ddi(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["drug1", "drug2", "interaction"],
        dtype={"drug1": "Int64", "drug2": "Int64", "interaction": "Int64"},
    )
    for c in ("drug1", "drug2", "interaction"):
        df[c] = df[c].astype(int, copy=False)
    return df


def assert_ddi_types(ddi: pd.DataFrame) -> None:
    assert pd.api.types.is_integer_dtype(ddi["drug1"])
    assert pd.api.types.is_integer_dtype(ddi["drug2"])
    assert pd.api.types.is_integer_dtype(ddi["interaction"])


class OutPaths:
    def __init__(self, save_root: str):
        self.root = save_root
        ensure_dir(self.root)

    def case_root(self, case: str) -> str:
        p = os.path.join(self.root, case)
        ensure_dir(p)
        return p

    def stats_dir(self, case: str, stage: Optional[str] = None) -> str:
        base = os.path.join(self.case_root(case), "stats")
        if stage:
            base = os.path.join(base, stage)
        ensure_dir(base)
        return base

    def viz_dir(self, case: str, stage: Optional[str] = None) -> str:
        base = os.path.join(self.case_root(case), "viz")
        if stage:
            base = os.path.join(base, stage)
        ensure_dir(base)
        return base

    def figs_dir(self, case: str, sub: Optional[str] = None) -> str:
        base = os.path.join(self.case_root(case), "figs")
        if sub:
            base = os.path.join(base, sub)
        ensure_dir(base)
        return base
