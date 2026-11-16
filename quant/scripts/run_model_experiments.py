"""Run resource-aware LightGBM, SVM and LSTM experiments on feature pkl slices."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR

try:  # pragma: no cover - optional in base env, present in quant env
    import lightgbm as lgb
except Exception:  # pragma: no cover
    lgb = None

try:  # pragma: no cover
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # pragma: no cover
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_DATA_DIR = PROJECT_ROOT / "ashare_minute_data_2023_2026_04"

from quant_backtest.features import feature_columns


class SmallLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(self.dropout(out[:, -1, :])).squeeze(-1)


def load_feature_data(input_dir: Path, max_files: int, max_rows: int) -> pd.DataFrame:
    files = sorted(input_dir.rglob("*.pkl"))
    if max_files:
        files = files[:max_files]
    parts: list[pd.DataFrame] = []
    rows = 0
    for path in files:
        part = pd.read_pickle(path)
        parts.append(part)
        rows += len(part)
        if max_rows and rows >= max_rows:
            break
    if not parts:
        raise FileNotFoundError(f"No feature pkl files found in {input_dir}")
    data = pd.concat(parts, ignore_index=True)
    if max_rows and len(data) > max_rows:
        data = data.iloc[:max_rows].copy()
    data["datetime"] = pd.to_datetime(data["datetime"])
    data["symbol"] = data["symbol"].astype(str)
    return data.sort_values(["datetime", "symbol"]).reset_index(drop=True)


def add_cross_sectional_target(data: pd.DataFrame) -> pd.DataFrame:
    frame = data.copy()
    count = frame.groupby("datetime")["symbol"].transform("count")
    rank = frame.groupby("datetime")["Return_24"].rank(method="average", pct=True)
    frame["target_rank"] = np.where(count > 1, rank, np.nan)
    return frame.dropna(subset=["Return_24", "target_rank"])


def split_by_time(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    times = pd.Index(sorted(data["datetime"].unique()))
    if len(times) < 10:
        raise ValueError("Not enough timestamps for time split.")
    train_end = times[int(len(times) * 0.6)]
    val_end = times[int(len(times) * 0.8)]
    train = data[data["datetime"] < train_end].copy()
    val = data[(data["datetime"] >= train_end) & (data["datetime"] < val_end)].copy()
    test = data[data["datetime"] >= val_end].copy()
    return train, val, test


def prepare_xy(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cols = feature_columns(train)
    med = train[cols].replace([np.inf, -np.inf], np.nan).median(numeric_only=True).fillna(0)

    def x(frame: pd.DataFrame) -> np.ndarray:
        values = frame[cols].replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0)
        return values.to_numpy(dtype=np.float32)

    return (
        cols,
        x(train),
        train["target_rank"].to_numpy(dtype=np.float32),
        x(val),
        val["target_rank"].to_numpy(dtype=np.float32),
        x(test),
        test["target_rank"].to_numpy(dtype=np.float32),
    )


def mean_spearman_ic(frame: pd.DataFrame, pred_col: str = "pred") -> float:
    values: list[float] = []
    for _, group in frame.groupby("datetime", sort=False):
        if group[pred_col].nunique() < 2 or group["Return_24"].nunique() < 2:
            continue
        corr = group[[pred_col, "Return_24"]].corr(method="spearman").iloc[0, 1]
        if pd.notna(corr):
            values.append(float(corr))
    return float(np.mean(values)) if values else 0.0


def long_short_metrics(frame: pd.DataFrame, pred_col: str = "pred", q: float = 0.2) -> dict[str, float]:
    returns: list[float] = []
    for _, group in frame.groupby("datetime", sort=False):
        if len(group) < 5:
            continue
        n = max(1, int(len(group) * q))
        ordered = group.sort_values(pred_col)
        short_ret = ordered.head(n)["Return_24"].mean()
        long_ret = ordered.tail(n)["Return_24"].mean()
        returns.append(float(long_ret - short_ret))
    if not returns:
        return {"ls_mean": 0.0, "ls_std": 0.0, "ls_sharpe_like": 0.0, "ls_periods": 0}
    arr = np.asarray(returns, dtype=float)
    sharpe_like = arr.mean() / arr.std(ddof=0) * math.sqrt(252 * 240) if arr.std(ddof=0) else 0.0
    return {
        "ls_mean": float(arr.mean()),
        "ls_std": float(arr.std(ddof=0)),
        "ls_sharpe_like": float(sharpe_like),
        "ls_periods": int(len(arr)),
    }


def evaluate(name: str, test: pd.DataFrame, preds: np.ndarray) -> dict[str, float | str | int]:
    out = test[["datetime", "symbol", "Return_24", "target_rank"]].copy()
    out["pred"] = preds
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["Return_24", "target_rank", "pred"])
    if out.empty:
        return {"model": name, "error": "no finite predictions"}
    metrics: dict[str, float | str | int] = {
        "model": name,
        "rmse_rank": float(mean_squared_error(out["target_rank"], out["pred"]) ** 0.5),
        "mae_rank": float(mean_absolute_error(out["target_rank"], out["pred"])),
        "mean_spearman_ic": mean_spearman_ic(out),
    }
    metrics.update(long_short_metrics(out))
    return metrics


def make_sequences(frame: pd.DataFrame, cols: list[str], window: int, max_sequences: int) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    xs: list[np.ndarray] = []
    ys: list[float] = []
    rows: list[pd.Series] = []
    for _, group in frame.groupby("symbol", sort=False):
        group = group.sort_values("datetime")
        values = group[cols].to_numpy(dtype=np.float32)
        targets = group["target_rank"].to_numpy(dtype=np.float32)
        for i in range(window - 1, len(group)):
            if np.isnan(targets[i]):
                continue
            xs.append(values[i - window + 1 : i + 1])
            ys.append(float(targets[i]))
            rows.append(group.iloc[i][["datetime", "symbol", "Return_24", "target_rank"]])
            if max_sequences and len(xs) >= max_sequences:
                return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32), pd.DataFrame(rows)
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32), pd.DataFrame(rows)


def run_lstm(train: pd.DataFrame, test: pd.DataFrame, cols: list[str], args: argparse.Namespace, output_dir: Path) -> tuple[dict[str, float | str | int], Path | None]:
    if torch is None:
        return {"model": "lstm", "error": "torch is not installed"}, None
    med = train[cols].replace([np.inf, -np.inf], np.nan).median(numeric_only=True).fillna(0)
    scaler = StandardScaler()
    train = train.copy()
    test = test.copy()
    train_values = train[cols].replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0).to_numpy(dtype=np.float32)
    test_values = test[cols].replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0).to_numpy(dtype=np.float32)
    train[cols] = np.nan_to_num(scaler.fit_transform(train_values), nan=0.0, posinf=0.0, neginf=0.0)
    test[cols] = np.nan_to_num(scaler.transform(test_values), nan=0.0, posinf=0.0, neginf=0.0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    x_train, y_train, _ = make_sequences(train, cols, args.lstm_window, args.lstm_max_sequences)
    x_test, y_test, test_rows = make_sequences(test, cols, args.lstm_window, args.lstm_max_sequences)
    if len(x_train) == 0 or len(x_test) == 0:
        return {"model": "lstm", "error": "not enough sequences"}, None

    model = SmallLSTM(input_size=len(cols), hidden_size=args.lstm_hidden).to(device)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=args.batch_size,
        shuffle=True,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    model.train()
    for epoch in range(args.lstm_epochs):
        losses: list[float] = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            if not torch.isfinite(loss):
                return {"model": "lstm", "error": "non-finite training loss"}, None
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        print(f"lstm epoch={epoch + 1}/{args.lstm_epochs} loss={np.mean(losses):.6f}")

    model.eval()
    preds: list[np.ndarray] = []
    test_loader = DataLoader(TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test)), batch_size=args.batch_size)
    with torch.no_grad():
        for xb, _ in test_loader:
            preds.append(model(xb.to(device)).detach().cpu().numpy())
    pred = np.concatenate(preds)
    metrics = evaluate("lstm", test_rows.reset_index(drop=True), pred)
    model_path = output_dir / "lstm_model.pt"
    torch.save({"state_dict": model.state_dict(), "feature_columns": cols, "window": args.lstm_window}, model_path)
    return metrics, model_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-files", type=int, default=8)
    parser.add_argument("--max-rows", type=int, default=250_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lstm-window", type=int, default=24)
    parser.add_argument("--lstm-hidden", type=int, default=64)
    parser.add_argument("--lstm-epochs", type=int, default=3)
    parser.add_argument("--lstm-max-sequences", type=int, default=120_000)
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir
    input_dir = args.input_dir or data_dir / "processed" / "features_pkl"
    output_dir = args.output_dir or PROJECT_ROOT / "reports" / "model_experiments"
    output_dir.mkdir(parents=True, exist_ok=True)

    data = add_cross_sectional_target(load_feature_data(input_dir, args.max_files, args.max_rows))
    train, val, test = split_by_time(data)
    cols, x_train, y_train, x_val, y_val, x_test, y_test = prepare_xy(train, val, test)
    print(f"rows train={len(train)} val={len(val)} test={len(test)} features={len(cols)}")

    results: list[dict[str, float | str | int]] = []
    artifacts: dict[str, str] = {"feature_count": str(len(cols))}

    svm = make_pipeline(StandardScaler(), LinearSVR(C=0.2, epsilon=0.01, max_iter=5000, random_state=7))
    svm.fit(x_train, y_train)
    svm_pred = svm.predict(x_test)
    results.append(evaluate("linear_svm", test, svm_pred))
    svm_path = output_dir / "linear_svm.joblib"
    joblib.dump({"model": svm, "feature_columns": cols}, svm_path)
    artifacts["linear_svm"] = str(svm_path)

    if lgb is not None:
        model = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=63,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=7,
            n_jobs=-1,
        )
        model.fit(
            x_train,
            y_train,
            eval_set=[(x_val, y_val)],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        lgb_pred = model.predict(x_test)
        results.append(evaluate("lightgbm", test, lgb_pred))
        lgb_path = output_dir / "lightgbm.joblib"
        joblib.dump({"model": model, "feature_columns": cols}, lgb_path)
        artifacts["lightgbm"] = str(lgb_path)

    lstm_metrics, lstm_path = run_lstm(train, test, cols, args, output_dir)
    results.append(lstm_metrics)
    if lstm_path is not None:
        artifacts["lstm"] = str(lstm_path)

    metrics = pd.DataFrame(results)
    metrics.to_csv(output_dir / "model_metrics.csv", index=False, encoding="utf-8-sig")
    (output_dir / "artifacts.json").write_text(json.dumps(artifacts, ensure_ascii=False, indent=2), encoding="utf-8")
    print(metrics)
    print(f"Saved metrics to {output_dir / 'model_metrics.csv'}")


if __name__ == "__main__":
    main()
