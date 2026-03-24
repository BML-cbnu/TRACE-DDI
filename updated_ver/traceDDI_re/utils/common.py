from __future__ import annotations

import logging
import os
import random

import numpy as np
import pytorch_lightning as pl
import torch

try:
    import pynvml  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    pynvml = None  # type: ignore[assignment]

logger = logging.getLogger("TRACE-DDI")


def set_seeds(seed: int) -> None:
    # Reproducibility
    pl.seed_everything(seed, workers=True)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dirs(*paths: str) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)


def init_nvml_if_available() -> bool:
    if pynvml is None or not torch.cuda.is_available():
        return False
    try:
        pynvml.nvmlInit()
        return True
    except Exception as e:
        logger.warning("NVML init failed (ignored): %s", e)
        return False


def shutdown_nvml(enabled: bool) -> None:
    if not enabled or pynvml is None:
        return
    try:
        pynvml.nvmlShutdown()
    except Exception:
        pass
