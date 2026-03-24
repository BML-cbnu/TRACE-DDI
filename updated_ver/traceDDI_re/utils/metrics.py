from __future__ import annotations

from typing import Any, Dict, Sequence, cast

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
)


def classification_report_dict(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    digits: int = 4,
    zero_division: int | str = 1,
) -> Dict[str, Any]:
    rep_any = classification_report(
        list(y_true),
        list(y_pred),
        output_dict=True,
        digits=digits,
        zero_division=cast(Any, zero_division),
    )
    return cast(Dict[str, Any], rep_any)


def classification_report_str(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    digits: int = 4,
    zero_division: int | str = 1,
) -> str:
    rep_any = classification_report(
        list(y_true),
        list(y_pred),
        output_dict=False,
        digits=digits,
        zero_division=cast(Any, zero_division),
    )
    return rep_any if isinstance(rep_any, str) else str(rep_any)


def compute_binary_metrics(
    y_true: np.ndarray,
    y_prob_pos: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob_pos = np.asarray(y_prob_pos, dtype=np.float64)

    y_pred = (y_prob_pos >= threshold).astype(np.int64)
    acc = float((y_pred == y_true).mean())

    out: Dict[str, float] = {"acc": acc}
    if len(np.unique(y_true)) == 2:
        out["auroc"] = float(roc_auc_score(y_true, y_prob_pos))
        out["auprc"] = float(average_precision_score(y_true, y_prob_pos))
    else:
        out["auroc"] = float("nan")
        out["auprc"] = float("nan")
    return out
