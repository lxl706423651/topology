"""三板块CTA回归检查：合成报价验证T+1、数量、止损路径及训练时间边界，不下载行情。"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from quant_backtest.cta_research_data import (classify_board, create_minute_clock, calculate_daily_context,
    select_history_universe, prepare_data, build_symbol_features)
from quant_backtest.cta_research_engine import (calculate_quantity, update_exit, create_parameters,
    simulate_path_labels, run_portfolio)
from quant_backtest.cta_research_reporting import audit_result
from quant_backtest.cta_research_training import create_folds, fit_model, XgbCtaStrategy, calculate_entry_mask, select_label_rows

ROOT = Path(__file__).resolve().parents[1]


def test_local_subset_selection_is_explicit_and_not_outcome_based(tmp_path):
    """本地模式只检查文件可用性，不能用完整未来状态/收益选股；默认模式仍要求全部股票。"""
    symbols = ["SH.600000", "SH.688001", "SZ.300001"]
    (tmp_path / "SH.688001.pkl").touch()
    selected, scope = select_history_universe(symbols, tmp_path,
        {"history_coverage_policy": "available_local", "history_data_dir": str(tmp_path)})
    assert selected == ["SH.688001"] and scope["eligible_raw_count"] == 3
    assert scope["excluded_symbols"] == ["SH.600000", "SZ.300001"]
    assert select_history_universe(symbols, tmp_path, {})[0] == symbols
    with pytest.raises(ValueError, match="显式指定"):
        select_history_universe(symbols, tmp_path, {"history_coverage_policy": "available_local"})


def test_offline_missing_benchmark_never_requests_network(cfg, tmp_path, monkeypatch):
    """cfg本地模式遇到必需基准缺失时停止，monkeypatch禁止任何下载调用。"""
    raw = tmp_path / "raw" / "2023"
    raw.mkdir(parents=True)
    (raw / "SH.600000.csv").touch()
    history = tmp_path / "history"
    history.mkdir()
    pd.DataFrame({"date": ["2023-01-02"], **{k: [1] for k in
        ["open", "high", "low", "close", "preclose", "volume", "amount", "tradestatus", "isST"]}}).to_pickle(history / "SH.600000.pkl")
    def reject_network(*args, **kwargs):
        """任何args/kwargs调用均代表离线契约失效。"""
        raise AssertionError("离线模式不允许网络调用")
    monkeypatch.setattr("quant_backtest.cta_research_data.download_history", reject_network)
    cfg.update(data_root=str(raw.parent), cache_dir=str(tmp_path / "cache"),
        history_data_dir=str(history), history_coverage_policy="available_local")
    with pytest.raises(PermissionError, match="历史数据尚缺1只"):
        prepare_data(cfg)


@pytest.mark.parametrize("trade_status", [0, np.nan])
def test_non_normal_daily_status_blocks_positive_minute_activity(cfg, tmp_path, monkeypatch, trade_status):
    """日trade_status非正常或未知时，即使分钟有成交也不能生成入场特征或可执行报价。"""
    date = pd.Timestamp("2024-01-02")
    times = create_minute_clock(pd.DatetimeIndex([date]))
    raw = pd.DataFrame({"datetime": times, "open": 100.0, "high": 101.0, "low": 99.0,
                        "close": 100.0, "volume": 10000, "amount": 10000000.0})
    history = pd.DataFrame({"date": pd.date_range("2023-08-01", date, freq="B"),
                           "high": 101.0, "low": 99.0, "close": 100.0, "preclose": 100.0,
                           "isST": 0, "tradestatus": 1.0})
    history.loc[history.index[-1], "tradestatus"] = trade_status
    monkeypatch.setattr("quant_backtest.cta_research_data.load_symbol_bars", lambda *args: (raw.copy(), {}))
    monkeypatch.setattr("quant_backtest.cta_research_data.read_history", lambda *args: history.copy())
    cfg.update(history_path=str(tmp_path), reuse_validated=False)
    output = build_symbol_features(("SH.600000", cfg, str(tmp_path), pd.DatetimeIndex([date]), [date]))
    payload = pd.read_pickle(output)
    assert payload["features"].empty
    assert not payload["quotes"].allow_buy.any() and not payload["quotes"].status_known.any()
    assert payload["audit"]["non_normal_active_minutes"] == 240


def test_reuse_cta_raw_features_not_old_context(cfg, tmp_path, monkeypatch):
    """cfg指向CTA分股票缓存时复用原因子，日尺度上下文重新生成，不能搬用旧池截面秩。"""
    date = pd.Timestamp("2024-01-02")
    times = create_minute_clock(pd.DatetimeIndex([date]))
    raw = pd.DataFrame({"datetime": times, "open": 100.0, "high": 101.0, "low": 99.0,
                        "close": 100.0, "volume": 10000, "amount": 10000000.0})
    history = pd.DataFrame({"date": pd.date_range("2023-08-01", date, freq="B"),
                           "high": 101.0, "low": 99.0, "close": 100.0, "preclose": 100.0,
                           "isST": 0, "tradestatus": 1.0})
    prior = tmp_path / "prior" / "symbols"
    prior.mkdir(parents=True)
    old = pd.DataFrame({"datetime": times, "symbol": "SH.600000", "momentum_5": 0.123,
                        "ctx_momentum5": 999, "cs_rank_momentum_5": 999})
    pd.to_pickle({"features": old}, prior / "SH.600000.pkl")
    monkeypatch.setattr("quant_backtest.cta_research_data.load_symbol_bars", lambda *args: (raw.copy(), {}))
    monkeypatch.setattr("quant_backtest.cta_research_data.read_history", lambda *args: history.copy())
    def reject_rebuild(*args):
        """已有全部时点时不得再次计算args所指的原因子。"""
        raise AssertionError("原因子被重复计算")
    monkeypatch.setattr("quant_backtest.cta_research_data.build_factor_library", reject_rebuild)
    cfg.update(history_path=str(tmp_path), reuse_validated=True, reuse_layout="cta", reuse_factor_cache=str(prior.parent))
    output = build_symbol_features(("SH.600000", cfg, str(tmp_path / "new"), pd.DatetimeIndex([date]), [date]))
    payload = pd.read_pickle(output)
    assert payload["audit"]["factors_reused"]
    assert np.allclose(payload["features"].momentum_5, 0.123)
    assert payload["features"].ctx_momentum5.eq(0).all()
    assert "cs_rank_momentum_5" not in payload["features"]


@pytest.fixture
def cfg():
    """读取实际YAML并缩小模型参数；不改变被测试的费用、数量及T+1约束。"""
    return yaml.safe_load((ROOT / "config/xgb_cta_walk_forward.yaml").read_text())


def create_market(days=4):
    """创建days个合成交易日、三个板块的只读等价报价对象，用于确定性撮合反例。"""
    times = create_minute_clock(pd.date_range("2024-01-02", periods=days, freq="B"))
    quotes = np.tile(np.array([100.0, 100.5, 99.5, 10000000.0, 2.0, 90.0, 110.0, 90.0]), (len(times), 3, 1))
    quotes[:, 2, 6:8] = [120.0, 80.0]
    flags = np.ones((len(times), 3, 3), dtype=np.uint8)
    flags[:, :, 2] = 0
    symbols = ["SH.600000", "SZ.300001", "SH.688001"]
    return SimpleNamespace(times=times, quotes=quotes, flags=flags, boards=np.array([0, 1, 2]),
                           symbols=symbols, symbol_index={s: i for i, s in enumerate(symbols)})


def create_signals(market, entries):
    """用entries=(分钟索引,股票列号,退出方案)生成固定正分数信号，不读取未来价格。"""
    return pd.DataFrame([{"datetime": market.times[t], "symbol": market.symbols[a], "prediction": 0.01,
                          "exit_profile": profile} for t, a, profile in entries],
                        columns=["datetime", "symbol", "prediction", "exit_profile"])


@pytest.mark.parametrize("symbol,board", [("SH.688001", 2), ("SZ.301001", 1), ("SZ.000001", 0),
                                          ("BJ.920001", -1), ("SH.689009", -1), ("SH.900901", -1)])
def test_board_classification(symbol, board):
    assert classify_board(symbol) == board


def test_star_order_sizes():
    assert calculate_quantity(1000, 19900, 100, 1e6, 2, True) == 0
    assert calculate_quantity(1000, 20100, 100, 1e6, 2, True) == 201
    assert calculate_quantity(1000, 20100, 100, 1e6, 1, True) == 200
    assert calculate_quantity(199, 1e6, 100, 1e6, 2, False) == 199
    assert calculate_quantity(199, 1e6, 100, 19800, 2, False) == 0
    assert calculate_quantity(600, 1e6, 100, 19900, 2, False) == 0


def test_independent_status_audit_rejects_false_execution_flags(tmp_path):
    """对tmp_path合成日线，独立审计必须发现已标停牌但分钟仍可执行的篡改。"""
    from scripts.audit_xgb_cta_results import audit_market_status
    market = create_market()
    for symbol in market.symbols:
        frame = pd.DataFrame({"date": market.times[::240].normalize(), "open": 100, "high": 101,
                             "low": 99, "close": 100, "preclose": 100, "volume": 1000,
                             "amount": 100000, "tradestatus": [1, 0, 1, 1], "isST": 0})
        frame.to_pickle(tmp_path / f"{symbol}.pkl")
    with pytest.raises(AssertionError, match="仍有交易资格"):
        audit_market_status(market, {"history_path": str(tmp_path)})
    market.flags[240:480, :, :2] = 0
    result = audit_market_status(market, {"history_path": str(tmp_path)})
    assert result["passed"] and result["blocked_active_minutes"] == 720


def test_t1_pending_stop_and_label_match(cfg):
    market = create_market()
    market.quotes[2:240, 2, 0:3] = [95.0, 95.5, 94.5]
    signals = create_signals(market, [(0, 2, 0)])
    result = run_portfolio(signals, market, market.times[0], market.times[-1] + pd.Timedelta(minutes=1), cfg)
    assert len(result["events"]) == 2
    assert result["events"].iloc[1].time_index == 240
    assert result["blocked"]["T+1尚不可卖"] > 0
    assert audit_result(result, market, cfg)["passed"]
    parameters, profiles, stamps = create_parameters(cfg, market.times)
    values, ends = simulate_path_labels(market.quotes, market.flags, market.boards, np.array([0]), np.array([2]),
                                        profiles[:1], parameters, stamps)
    event = result["events"].iloc[0]
    actual = result["trips"].net_pnl.sum() / (event.notional + event.fee)
    assert ends[0, 0] == 240
    assert values[0, 0] == pytest.approx(actual, abs=1e-12)


@pytest.mark.parametrize("condition", ["st", "upper", "lower", "missing", "capacity"])
def test_blocked_buys(cfg, condition):
    market = create_market()
    if condition == "st":
        market.flags[1, 2] = [0, 1, 1]
    elif condition == "upper":
        market.quotes[1, 2, 0] = 120
    elif condition == "lower":
        market.quotes[1, 2, 0] = 80
    elif condition == "missing":
        market.quotes[1, 2, 0] = np.nan
        market.flags[1, 2, :2] = 0
    else:
        market.quotes[1, 2, 3] = 1000
    result = run_portfolio(create_signals(market, [(0, 2, 0)]), market, market.times[0], market.times[-1], cfg)
    assert result["events"].empty
    assert len(result["blocked_events"]) == 1
    assert audit_result(result, market, cfg)["passed"]


def test_same_day_addition_blocked_and_next_day_allowed(cfg):
    market = create_market()
    entries = [(0, 2, 1), (24, 2, 1), (240, 2, 1)]
    result = run_portfolio(create_signals(market, entries), market, market.times[0], market.times[-1], cfg)
    buys = result["events"].query("side == 'buy'")
    assert buys.time_index.tolist() == [1, 241]
    assert buys.lot_id.nunique() == 2
    assert audit_result(result, market, cfg)["passed"]


def test_stop_only_tightens_and_excludes_unknown_intrabar_order():
    quote = np.array([105, 110, 100, 1e6, 2, 90, 120, 80], dtype=float)
    high, stop, reason = update_exit(quote, False, 100, 95, 100, 2, 2, np.array([2, 3, 10, 1]))
    assert high == 110 and stop == 104 and reason == 0
    quote[4] = 10
    _, widened, _ = update_exit(quote, False, high, stop, 100, 2, 3, np.array([2, 3, 10, 1]))
    assert widened == 104


def test_partial_sell_and_capacity(cfg):
    market = create_market()
    market.quotes[2:, 2, 0:3] = [95.0, 95.5, 94.5]
    market.quotes[240:245, 2, 3] = 500000
    result = run_portfolio(create_signals(market, [(0, 2, 0)]), market, market.times[0], market.times[-1], cfg)
    sells = result["events"].query("side == 'sell'")
    assert len(sells) > 1
    assert result["metrics"]["closed_lot_count"] == 1
    assert audit_result(result, market, cfg)["passed"]


def test_future_quote_changes_do_not_change_past_trades(cfg):
    market = create_market()
    signals = create_signals(market, [(0, 0, 1), (240, 2, 1)])
    first = run_portfolio(signals, market, market.times[0], market.times[-1], cfg)
    market.quotes[480:, :, :3] *= 0.9
    second = run_portfolio(signals, market, market.times[0], market.times[-1], cfg)
    pd.testing.assert_frame_equal(first["events"].query("time_index < 480"), second["events"].query("time_index < 480"))
    pd.testing.assert_frame_equal(first["curve"].iloc[:480], second["curve"].iloc[:480])


def test_daily_features_do_not_use_current_days_close(cfg):
    dates = pd.date_range("2023-01-01", periods=90)
    history = pd.DataFrame({"date": dates, "close": np.arange(90) + 100.0, "high": np.arange(90) + 101.0,
                            "low": np.arange(90) + 99.0, "preclose": np.arange(90) + 99.0, "isST": 0, "tradestatus": 1})
    before = calculate_daily_context(history, cfg, 2)
    history.loc[89, ["close", "high", "low"]] = [2000, 2200, 1900]
    after = calculate_daily_context(history, cfg, 2)
    pd.testing.assert_series_equal(before.iloc[-1], after.iloc[-1])


def test_folds_strictly_forward(cfg):
    folds = create_folds(cfg)
    assert folds[0]["test_start"] == pd.Timestamp("2023-10-01")
    for fold in folds:
        assert fold["train_end"] == fold["validation_start"] < fold["test_start"]
        assert fold["validation_end"] == fold["test_start"]
    assert folds[-1]["test_end"] == pd.Timestamp("2026-05-01")


def test_training_purges_crossing_labels(cfg, tmp_path):
    rng = np.random.default_rng(4)
    dates = pd.date_range("2023-01-01", periods=180)
    panel = pd.DataFrame({"datetime": dates, "symbol": "SH.600000", "cs_rank_momentum_5": rng.random(180),
                          "ctx_momentum5": rng.random(180)})
    labels = pd.DataFrame({"net_return_0": rng.normal(0, 0.01, 180), "label_end_0": dates + pd.Timedelta(days=4)})
    labels.loc[10, "label_end_0"] = pd.Timestamp("2024-01-01")
    fold = {"year": 2024, "train_start": pd.Timestamp("2023-01-01"), "train_end": pd.Timestamp("2023-07-01")}
    fit_model(panel, labels, fold, 0, {"name": "test", "n_estimators": 2, "max_depth": 2}, cfg, tmp_path)
    rows = pd.read_csv(tmp_path / "training_rows.csv")
    assert 10 not in rows.row_id.to_list()
    assert pd.to_datetime(rows.label_end).max() < fold["train_end"]


def test_signal_adapter_ignores_future_columns(cfg):
    class ConstantModel:
        def predict(self, frame):
            return np.ones(len(frame))
    frame = pd.DataFrame({"datetime": pd.to_datetime(["2024-01-02 09:31", "2024-01-02 09:55"]),
                          "symbol": ["SH.688001", "SH.688001"], "ctx_momentum20": 0.1,
                          "ctx_ema20_gap": 0.1, "ctx_momentum5": 0.03, "ctx_ema5_gap": 0.01,
                          "feature": [0.3, 0.8], "future_return": [1.0, -1.0]})
    rule = {"style": "trend", "entry_minutes": 24, "gate": 0.003, "exit_profile": 0}
    strategy = XgbCtaStrategy(ConstantModel(), ["feature"], rule, cfg)
    first = strategy.generate_signals(frame)
    frame["future_return"] *= -100
    pd.testing.assert_frame_equal(first, strategy.generate_signals(frame))
    assert first.open_signal.tolist() == [1, 1]
    assert first.close_signal.eq(0).all()


def test_label_sampling_independent_of_universe_column_numbers(cfg):
    times = create_minute_clock(pd.date_range("2023-01-03", periods=4, freq="B"))[::24]
    original = pd.DataFrame({"datetime": times, "symbol": "SH.600010"})
    extended = pd.concat([original, original.assign(symbol="SH.600000")], ignore_index=True)
    extended = extended.sort_values(["datetime", "symbol"]).reset_index(drop=True)
    left = original.iloc[select_label_rows(original, cfg)].datetime.tolist()
    selected = extended.iloc[select_label_rows(extended, cfg)]
    right = selected.loc[selected.symbol.eq("SH.600010"), "datetime"].tolist()
    assert left == right and len(left) == 8


def test_missing_prior_quote_does_not_remove_sell_protection(cfg):
    market = create_market()
    market.quotes[2:240, 2, 0:3] = [95.0, 95.5, 94.5]
    market.quotes[239, 2, 0:3] = np.nan
    market.flags[239, 2, :2] = 0
    result = run_portfolio(create_signals(market, [(0, 2, 0)]), market, market.times[0], market.times[-1], cfg)
    # 次日100较最后已知95高5%以上，卖价向有利方向变化仍在卖出保护范围内，可以卖出。
    assert result["events"].iloc[1].time_index == 240
    assert audit_result(result, market, cfg)["passed"]


def test_daily_valuation_tamper_is_rejected(cfg):
    market = create_market()
    result = run_portfolio(create_signals(market, [(0, 2, 0)]), market, market.times[0], market.times[-1], cfg)
    result["curve"].loc[239, "equity"] += 100
    with pytest.raises(AssertionError, match="日终净值"):
        audit_result(result, market, cfg)
