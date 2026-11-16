"""Strategy interfaces and adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .modeling import load_model

SignalFrame = pd.DataFrame


class BaseStrategy:
    """Base class for strategies.

    Implement ``generate_signals`` and return either:
    - ``target_position``: desired position in [-1, 1], or
    - ``open_signal`` and ``close_signal``: notebook-compatible signals.
    """

    name = "base"

    def generate_signals(self, data: pd.DataFrame) -> SignalFrame:
        raise NotImplementedError


@dataclass
class FunctionSignalStrategy(BaseStrategy):
    """Wrap old notebook functions shaped as ``cal_feature(data, paras)``."""

    func: Callable[[pd.DataFrame, dict[str, Any]], tuple[pd.Series, pd.Series]]
    paras: dict[str, Any] = field(default_factory=dict)
    name: str = "function_signal_strategy"

    def generate_signals(self, data: pd.DataFrame) -> SignalFrame:
        open_signal, close_signal = self.func(data.copy(), self.paras)
        return pd.DataFrame(
            {
                "open_signal": pd.Series(open_signal, index=data.index).fillna(0).astype(float),
                "close_signal": pd.Series(close_signal, index=data.index).fillna(0).astype(float),
            },
            index=data.index,
        )


@dataclass
class ModelSignalStrategy(BaseStrategy):
    """Generic model strategy for packaged ML/deep-learning models."""

    model: Any | None = None
    model_path: str | Path | None = None
    feature_fn: Callable[[pd.DataFrame], pd.DataFrame] | None = None
    threshold: float = 0.0
    long_label: int | float = 1
    short_label: int | float = -1
    flat_label: int | float = 0
    name: str = "model_signal_strategy"

    def __post_init__(self) -> None:
        if self.model is None and self.model_path is not None:
            self.model = load_model(self.model_path)
        if self.model is None:
            raise ValueError("ModelSignalStrategy requires model or model_path.")

    def generate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.feature_fn is not None:
            return self.feature_fn(data.copy())
        numeric = data.select_dtypes(include=[np.number]).copy()
        return numeric.drop(columns=["target_position", "open_signal", "close_signal"], errors="ignore")

    def generate_signals(self, data: pd.DataFrame) -> SignalFrame:
        features = self.generate_features(data)
        features = features.replace([np.inf, -np.inf], np.nan).fillna(0)

        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(features)
            if proba.shape[1] == 2:
                score = proba[:, 1] - 0.5
                raw = np.where(score > self.threshold, self.long_label, self.flat_label)
                raw = np.where(score < -self.threshold, self.short_label, raw)
            else:
                classes = getattr(self.model, "classes_", np.arange(proba.shape[1]))
                raw = classes[np.argmax(proba, axis=1)]
        elif hasattr(self.model, "predict"):
            raw = self.model.predict(features)
        else:
            raw = self.model(features)

        raw = np.asarray(raw)
        if raw.ndim > 1:
            raw = raw.squeeze()
        position = pd.Series(raw, index=data.index).replace(
            {
                self.long_label: 1,
                self.short_label: -1,
                self.flat_label: 0,
            }
        )
        position = pd.to_numeric(position, errors="coerce").fillna(0).clip(-1, 1)
        return pd.DataFrame({"target_position": position}, index=data.index)
