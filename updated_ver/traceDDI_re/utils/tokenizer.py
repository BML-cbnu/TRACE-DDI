from __future__ import annotations

import re
from typing import Dict, List, Sequence, Set


class SMILESTokenizer:
    # Regex tokenizer
    def __init__(self, smiles_list: Sequence[str], case_sensitive: bool = True) -> None:
        self.special_tokens: Dict[str, int] = {
            "<PAD>": 0,
            "<UNK>": 1,
            "<CLS>": 2,
            "<SEP>": 3,
        }
        self.case_sensitive = case_sensitive
        self.vocab: Dict[str, int] = self._build_vocab(smiles_list)

    @staticmethod
    def _compile_pattern(case_sensitive: bool = True) -> re.Pattern[str]:
        flags = 0 if case_sensitive else re.I
        return re.compile(
            r"(%\d{2}|\[[^\[\]]*\]|Br|Cl|Si|Al|Na|K|Ca|Mg|Cu|Co|Zn|Fe|Mn|P|\.|=|#|-|\+|\(|\)|\[|\]|\{|\}|[A-Za-z]|\d+|@|\\|/)",
            flags,
        )

    def _build_vocab(self, smiles_list: Sequence[str]) -> Dict[str, int]:
        tokens: Set[str] = set()
        pat = self._compile_pattern(self.case_sensitive)
        for s in smiles_list:
            tokens.update(pat.findall(s))

        out: Dict[str, int] = dict(self.special_tokens)
        offset = len(self.special_tokens)
        for i, tok in enumerate(sorted(tokens)):
            out[tok] = i + offset
        return out

    def tokenize(self, smiles: str) -> List[int]:
        pat = self._compile_pattern(self.case_sensitive)
        unk = self.special_tokens["<UNK>"]
        return [self.vocab.get(tok, unk) for tok in pat.findall(smiles)]

    def encode(self, smiles: str, max_length: int) -> List[int]:
        cls_id = self.special_tokens["<CLS>"]
        pad_id = self.special_tokens["<PAD>"]

        ids = [cls_id] + self.tokenize(smiles)
        if len(ids) > max_length:
            ids = ids[:max_length]
        return ids + [pad_id] * (max_length - len(ids))
