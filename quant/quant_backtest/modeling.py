"""Optional model loading helpers for machine-learning strategies."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import joblib


def load_model(path: str | Path) -> Any:
    """Load a packaged model without making heavyweight ML libs mandatory.

    Supported out of the box:
    - ``.joblib``: sklearn/lightgbm/xgboost style objects saved with joblib
    - ``.pkl``/``.pickle``: pickle-compatible Python objects
    - ``.pt``/``.pth``: torch models when torch is installed
    - ``.keras``/``.h5``: keras models when tensorflow is installed
    """

    model_path = Path(path)
    suffix = model_path.suffix.lower()
    if suffix == ".joblib":
        return joblib.load(model_path)
    if suffix in {".pkl", ".pickle"}:
        with model_path.open("rb") as handle:
            return pickle.load(handle)
    if suffix in {".pt", ".pth"}:
        try:
            import torch
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError("Loading .pt/.pth models requires torch to be installed.") from exc
        return torch.load(model_path, map_location="cpu")
    if suffix in {".keras", ".h5"}:
        try:
            from tensorflow import keras
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError("Loading keras models requires tensorflow to be installed.") from exc
        return keras.models.load_model(model_path)
    raise ValueError(f"Unsupported model file type: {model_path.suffix}")
