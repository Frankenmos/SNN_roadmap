from __future__ import annotations

from typing import Any

import numpy as np


def coerce_numeric_rows(rows: Any) -> np.ndarray | None:
    try:
        arr = np.asarray(rows)
    except Exception:
        return None
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim == 2 and arr.dtype != object:
        return arr
    return None


def safe_float(value: Any) -> float:
    try:
        return float(0.0 if value is None else value)
    except (TypeError, ValueError):
        return 0.0
