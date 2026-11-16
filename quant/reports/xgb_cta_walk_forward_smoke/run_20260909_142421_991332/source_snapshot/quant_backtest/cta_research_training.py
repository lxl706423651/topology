"""XGBoost CTA严格向前训练：标签、特征选择、规则选择与外层测试分离。

输入为扩展PKL/只读报价和YAML；输出原生模型、训练输入指纹、候选指标与冻结规则。
年度测试不参与拟合或调参，所有年度规则冻结之后才开始任何外层评价。
"""

from __future__ import annotations

from datetime import datetime
import itertools
import json
from pathlib import Path
import shutil
import subprocess
import zlib

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml

from .annual_panel import write_json
from .annual_training import describe_feature
from .cta_research_data import CONTEXT_NAMES, calculate_identity, calculate_sha256, save_pickle
from .cta_research_engine import create_parameters, simulate_path_labels, run_portfolio
from .strategy import BaseStrategy


def create_folds(cfg: dict) -> list[dict]:
    """按cfg生成向前折；2023只测试第四季度，其余年份测试全年或截至数据末日。"""
    folds = []
    for year in cfg["test_years"]:
        test_start = pd.Timestamp("2023-10-01" if year == 2023 else f"{year}-01-01")
        validation_start = pd.Timestamp("2023-07-01" if year == 2023 else f"{year - 1}-10-01")
        test_end = min(pd.Timestamp(f"{year + 1}-01-01"), pd.Timestamp(cfg["end_date"]) + pd.Timedelta(days=1))
        folds.append({"year": year, "train_start": pd.Timestamp(cfg["start_date"]), "train_end": validation_start,
                      "validation_start": validation_start, "validation_end": test_start,
                      "test_start": test_start, "test_end": test_end})
    if any(f["train_start"] >= f["train_end"] or f["test_start"] >= f["test_end"] for f in folds):
        raise ValueError("向前切分缺少训练或测试数据")
    return folds


def calculate_entry_mask(frame: pd.DataFrame, style: str, frequency: int) -> np.ndarray:
    """仅从frame当前/历史特征按style趋势条件及frequency交易分钟频率生成入场资格。"""
    minutes = frame.datetime.dt.hour * 60 + frame.datetime.dt.minute
    session = np.where(minutes <= 690, minutes - 571, minutes - 781 + 120)
    clock = (session % frequency == 0) & (session < 236)
    trend = (frame.ctx_momentum20 > 0) & (frame.ctx_ema20_gap > 0)
    if style == "trend":
        condition = trend & (frame.ctx_momentum5 > 0) & (frame.ctx_ema5_gap > 0)
    elif style == "breakout":
        condition = trend & (frame.ctx_breakout10 > 0)
    elif style == "pullback":
        condition = trend & frame.ctx_ema5_gap.between(0, 0.02) & frame.ctx_momentum5.between(-0.05, 0.05)
    else:
        raise ValueError(f"未定义CTA条件：{style}")
    return np.asarray(clock & condition)


class XgbCtaStrategy(BaseStrategy):
    """三板块只做多开仓适配器；rule指定频率/CTA/门槛，cfg限制前5与每笔10%。

    使用open_signal合同而非target_position，避免零信号被通用引擎误读为立刻清仓。
    平仓、分批加仓和T+1由专用独立批次引擎负责，不能直接接无T+1的演示引擎。
    """

    name = "xgb_cta_walk_forward"

    def __init__(self, model, features: list[str], rule: dict, cfg: dict):
        """model是过去拟合的XGBoost，features为固定列序，rule/cfg为冻结规则与仓位设置。"""
        self.model, self.features, self.rule, self.cfg = model, list(features), dict(rule), dict(cfg)

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """对data完成特征推断；保留原索引，不接收标签，不改变调用方DataFrame。"""
        output = data[["datetime", "symbol"]].copy()
        prediction = self.model.predict(data[self.features]) / 100.0
        eligible = calculate_entry_mask(data, self.rule["style"], self.rule["entry_minutes"])
        eligible &= np.isfinite(prediction) & (prediction > self.rule["gate"])
        output["prediction"] = prediction
        output["open_signal"] = eligible.astype(int)
        output["close_signal"] = 0
        output["exit_profile"] = self.rule["exit_profile"]
        return output


def load_labels(panel: pd.DataFrame, market, manifest: dict, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """为panel每日固定轮换的少量时间点计算路径标签，缓存依赖数据/引擎/退出配置。

    候选抽样只由股票、日期和时钟决定，不使用未来价格；标签只供训练，永不合入预测特征。
    """
    index = market.times.get_indexer(panel.datetime)
    assets = panel.symbol.map(market.symbol_index).to_numpy(dtype=np.int64)
    # 按股票与交易日错开两个时点，覆盖日内所有位置，避免只训练开盘点。
    row_ids = select_label_rows(panel, cfg)
    identity = {"panel": manifest["panel_sha256"], "profiles": cfg["exit_profiles"],
                "engine": calculate_sha256(Path(__file__).with_name("cta_research_engine.py")),
                "label_code": calculate_sha256(Path(__file__)), "rows": calculate_identity(row_ids.tolist()),
                "parameters": {k: cfg[k] for k in ["initial_cash", "entry_equity_fraction", "commission_rate", "min_commission",
                "transfer_rate", "slippage_rate", "max_bar_amount_fraction", "stamp_rate_before_20230828", "stamp_rate_from_20230828"]}}
    path = Path(manifest["cache_path"]) / f"path_labels_{calculate_identity(identity)}.pkl"
    if path.exists():
        print(f"复用执行路径标签：{path.name}", flush=True)
        return pd.read_pickle(path), identity
    parameters, profiles, stamps = create_parameters(cfg, market.times)
    pieces = []
    # 小批量让日志可见；Numba函数对输入只读，不使用全账户实际测试持仓构造标签。
    for begin in range(0, len(row_ids), 10000):
        rows = row_ids[begin:begin + 10000]
        values, ends = simulate_path_labels(market.quotes, market.flags, market.boards, index[rows], assets[rows],
                                             profiles, parameters, stamps)
        piece = pd.DataFrame({"row_id": rows})
        for profile in range(len(profiles)):
            piece[f"net_return_{profile}"] = values[:, profile]
            dates = np.full(len(rows), np.datetime64("NaT"), dtype="datetime64[ns]")
            valid = ends[:, profile] >= 0
            dates[valid] = market.times[ends[valid, profile]].to_numpy()
            piece[f"label_end_{profile}"] = dates
        pieces.append(piece)
        print(f"路径监督标签 {min(begin + 10000, len(row_ids)):,}/{len(row_ids):,}", flush=True)
    labels = pd.concat(pieces, ignore_index=True).set_index("row_id")
    save_pickle(path, labels)
    write_json(path.with_suffix(".json"), identity)
    return labels, identity


def select_label_rows(panel: pd.DataFrame, cfg: dict) -> np.ndarray:
    """按panel股票代码的稳定校验值和自然日期轮换时点；不依赖未来上市股票造成的列号变化。"""
    slots = 240 // cfg["base_signal_minutes"]
    codes = {symbol: zlib.crc32(symbol.encode("ascii")) for symbol in panel.symbol.unique()}
    days = panel.datetime.dt.normalize().astype("int64") // 86400000000000
    offset = (days + panel.symbol.map(codes)) % slots
    minutes = panel.datetime.dt.hour * 60 + panel.datetime.dt.minute
    session = np.where(minutes <= 690, minutes - 571, minutes - 781 + 120)
    selected = ((session // cfg["base_signal_minutes"] - offset) % slots) < cfg["labels_per_symbol_day"]
    return np.flatnonzero(selected)


def select_features(train: pd.DataFrame, target: pd.Series, cfg: dict) -> tuple[list[str], pd.DataFrame]:
    """只按train/target的覆盖率、相关性和冗余筛选cfg数量上限的特征，返回审计表。"""
    columns = [c for c in train if c.startswith(("cs_rank_", "ctx_"))]
    stride = max(1, len(train) // 40000)
    sample = train.iloc[::stride]
    coverage = train[columns].notna().mean()
    candidates = [c for c in columns if coverage[c] >= cfg["feature_min_coverage"] and sample[c].std() > 1e-8]
    strength = sample[candidates].corrwith(target.reindex(sample.index)).fillna(0)
    correlations = sample[candidates].corr().abs().fillna(0)
    ordered = sorted(candidates, key=lambda c: (-abs(strength[c]), c))
    selected, audit = [], []
    for column in ordered:
        redundancy = float(correlations.loc[column, selected].max()) if selected else 0.0
        keep = len(selected) < cfg["max_features"] and redundancy < cfg["feature_max_correlation"]
        audit.append({"feature": column, "说明": CONTEXT_NAMES.get(column, describe_feature(column)),
                      "coverage": coverage[column], "training_correlation": strength[column],
                      "maximum_redundancy": redundancy, "selected": keep})
        if keep:
            selected.append(column)
    for column in sorted(set(columns) - set(candidates)):
        audit.append({"feature": column, "说明": CONTEXT_NAMES.get(column, describe_feature(column)),
                      "coverage": coverage[column], "selected": False})
    if not selected:
        raise ValueError("本折没有合格的训练特征")
    return selected, pd.DataFrame(audit)


def fit_model(panel: pd.DataFrame, labels: pd.DataFrame, fold: dict, exit_profile: int, model_profile: dict,
              cfg: dict, directory: Path) -> tuple[object, list[str]]:
    """在fold核心过去区间拟合指定退出/模型方案，落盘输入行/原生权重，不访问外层预测。"""
    selected = labels.join(panel[["datetime", "symbol"]])
    end = selected[f"label_end_{exit_profile}"]
    target = selected[f"net_return_{exit_profile}"]
    mask = (selected.datetime >= fold["train_start"]) & (selected.datetime < fold["train_end"])
    mask &= (end < fold["train_end"]) & target.notna()
    selected = selected.loc[mask]
    if len(selected) < 100:
        raise ValueError(f"{fold['year']}核心训练标签不足100行")
    train = panel.loc[selected.index]
    target = selected[f"net_return_{exit_profile}"].clip(-0.5, 0.5) * 100.0
    features, audit = select_features(train, target, cfg)
    params = {k: v for k, v in model_profile.items() if k != "name"}
    model = xgb.XGBRegressor(**params, objective="reg:squarederror", tree_method="hist", learning_rate=0.04,
        subsample=0.85, colsample_bytree=0.85, reg_alpha=0.2, random_state=cfg["random_seed"], n_jobs=cfg["workers"])
    weights = 1.0 / train.groupby(train.datetime.dt.normalize()).symbol.transform("size")
    weights /= weights.mean()
    model.fit(train[features], target, sample_weight=weights)
    directory.mkdir(parents=True, exist_ok=True)
    model.save_model(directory / "xgboost_model.json")
    check = xgb.XGBRegressor()
    check.load_model(directory / "xgboost_model.json")
    sample = train[features].iloc[::max(1, len(train) // 1000)]
    if not np.array_equal(model.predict(sample), check.predict(sample)):
        raise AssertionError("原生模型重载后预测不一致")
    audit.to_csv(directory / "model_feature_manifest.csv", index=False, encoding="utf-8-sig")
    rows = train[["datetime", "symbol"]].copy()
    rows["label_end"] = selected[f"label_end_{exit_profile}"]
    rows["target_net_return"] = selected[f"net_return_{exit_profile}"]
    rows["sample_weight"] = weights
    rows.to_csv(directory / "training_rows.csv", index_label="row_id", encoding="utf-8-sig")
    write_json(directory / "model_metadata.json", {"fold": fold, "features": features, "model_profile": model_profile,
        "exit_profile": exit_profile, "label": "单笔10%初始资金预算，T+1、次分钟成交、独立ATR/趋势/到期退出、费用后的净收益；不是账户净收益",
        "fit_rows": len(train), "last_training_feature": train.datetime.max(), "last_training_label": rows.label_end.max(),
        "target_clip": [-0.5, 0.5], "target_scale": 100, "weights": "每个训练日总权重相同",
        "training_input_sha256": calculate_sha256(directory / "training_rows.csv"),
        "feature_values_sha256": calculate_identity(pd.util.hash_pandas_object(train[features], index=True).tolist()),
        "model_sha256": calculate_sha256(directory / "xgboost_model.json"), "xgboost_version": xgb.__version__,
        "actual_model_parameters": model.get_params(), "is_smoke": cfg.get("is_smoke", False),
        "native_reload_verified": True, "uses_future_years": False, "refit_after_rule_selection": False})
    return model, features


def select_rule(panel: pd.DataFrame, labels: pd.DataFrame, market, fold: dict, cfg: dict, directory: Path) -> dict:
    """仅用fold验证期比较预先固定的模型/CTA/频率/门槛，在读取外层之前冻结唯一方案。"""
    validation = panel.loc[(panel.datetime >= fold["validation_start"]) & (panel.datetime < fold["validation_end"])]
    rows, artifacts = [], {}
    for exit_id, model_profile in itertools.product(range(len(cfg["exit_profiles"])), cfg["model_profiles"]):
        model_id = f"exit{exit_id}_{model_profile['name']}"
        model_dir = directory / "models" / model_id
        model, features = fit_model(panel, labels, fold, exit_id, model_profile, cfg, model_dir)
        artifacts[model_id] = {"model_dir": str(model_dir), "features": features}
        scores = model.predict(validation[features]) / 100.0
        for style, frequency, gate in itertools.product(cfg["entry_styles"], cfg["entry_minutes_candidates"], cfg["net_return_gates"]):
            eligible = calculate_entry_mask(validation, style, frequency) & np.isfinite(scores) & (scores > gate)
            signals = validation.loc[eligible, ["datetime", "symbol"]].copy()
            signals["prediction"] = scores[eligible]
            signals["exit_profile"] = exit_id
            result = run_portfolio(signals, market, fold["validation_start"], fold["validation_end"], cfg)
            key = f"{model_id}_{style}_m{frequency}_gate{gate:g}"
            rows.append({"rule_id": key, "model_id": model_id, "style": style, "entry_minutes": frequency,
                         "gate": gate, "exit_profile": exit_id, **result["metrics"]})
        print(f"{fold['year']}内部：已拟合{model_id}，已评估{len(rows)}个规则", flush=True)
        pd.DataFrame(rows).to_csv(directory / "validation_rule_metrics.csv", index=False, encoding="utf-8-sig")
    table = pd.DataFrame(rows)
    eligible = table.loc[(table.closed_lot_count >= cfg["validation_min_closed_lots"]) & (table.annual_volatility > 0)]
    sufficient = len(eligible) > 0
    if not sufficient:
        # 不用额外收益开关伪造现金策略；固定降级顺序展示诊断，并禁止称其验证合格。
        eligible = table.sort_values(["closed_lot_count", "rule_id"], ascending=[False, True]).head(1)
    winner = eligible.sort_values(["sharpe", "max_drawdown", "rule_id"], ascending=[False, False, True]).iloc[0]
    keys = ["rule_id", "model_id", "style", "entry_minutes", "gate", "exit_profile"]
    selected = {k: winner[k].item() if hasattr(winner[k], "item") else winner[k] for k in keys}
    selected.update(artifacts[winner.model_id])
    selected.update(validation_sufficient=sufficient, validation_metrics={k: float(winner[k]) for k in ["sharpe", "total_return", "closed_lot_count"]},
                    selection="只用内部验证净日收益夏普；无足够交易时按完整批次数降序/规则ID升序展示未晋级诊断")
    write_json(directory / "selected_rule.json", selected)
    return selected


def generate_frozen_signals(panel: pd.DataFrame, selected: dict, fold: dict, cfg: dict) -> pd.DataFrame:
    """将selected模型/规则搬到fold测试期；只传特征列，不读取任何未来标签或测试收益。"""
    model = xgb.XGBRegressor()
    model.load_model(Path(selected["model_dir"]) / "xgboost_model.json")
    strategy = XgbCtaStrategy(model, selected["features"], selected, cfg)
    data = panel.loc[(panel.datetime >= fold["test_start"]) & (panel.datetime < fold["test_end"])]
    predicted = strategy.generate_signals(data)
    return predicted.loc[predicted.open_signal.eq(1), ["datetime", "symbol", "prediction", "exit_profile"]]


def run_walk_forward(panel: pd.DataFrame, membership: pd.DataFrame, market, manifest: dict, cfg: dict) -> Path:
    """以cfg冻结全部过去模型/规则后逐年测试，并另外执行跨年连续账户与事先声明的成本压力测试。"""
    from .cta_research_reporting import save_result, save_summary, audit_result
    root = Path(__file__).resolve().parents[1]
    directory = Path(cfg["output_dir"]) / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    directory.mkdir(parents=True, exist_ok=False)
    source_names = ["quant_backtest/cta_research_data.py", "quant_backtest/cta_research_engine.py",
        "quant_backtest/cta_research_training.py", "quant_backtest/cta_research_reporting.py", "quant_backtest/annual_portfolio.py",
        "quant_backtest/factor_library.py", "quant_backtest/annual_training.py", "quant_backtest/annual_panel.py",
        "quant_backtest/strategy.py", "scripts/run_xgb_cta_walk_forward.py", "config/xgb_cta_walk_forward.yaml",
        "scripts/run_dynamic_high_vol_method_backtests.py", "scripts/run_annual_method_validation.py"]
    sources = {}
    for name in source_names:
        source = root / name
        destination = directory / "source_snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        sources[name] = calculate_sha256(source)
    (directory / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    folds = create_folds(cfg)
    write_json(directory / "protocol.json", {"folds": folds, "sources": sources, "freeze_all_before_outer": True,
        "exposed_historical_data": True, "annual_accounts_reset_cash": True, "separate_continuous_account": True,
        "outer_target": "跨年连续账户净日收益夏普>3；不是各年度平均代替连续夏普",
        "fixed_stress_test": "已选规则与信号固定，滑点从万五翻倍为千一",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()})
    write_json(directory / "data_manifest.json", manifest)
    membership.to_csv(directory / "dynamic_universe_membership.csv", index=False, encoding="utf-8-sig")
    labels, identity = load_labels(panel, market, manifest, cfg)
    write_json(directory / "label_manifest.json", identity)
    selected_rules = []
    for fold in folds:
        fold_dir = directory / f"year_{fold['year']}"
        fold_dir.mkdir()
        print(f"开始{fold['year']}折，只用{fold['train_start'].date()}至{fold['train_end'].date()}之前拟合", flush=True)
        selected = select_rule(panel, labels, market, fold, cfg, fold_dir)
        selected_rules.append(selected)
    frozen = {}
    for path in directory.glob("year_*/**/*"):
        if path.is_file():
            frozen[str(path.relative_to(directory))] = calculate_sha256(path)
    write_json(directory / "frozen_before_test.json", {"files": frozen, "time": datetime.now().isoformat()})
    all_signals, annual = [], []
    for fold, selected in zip(folds, selected_rules):
        print(f"冻结模型后单次年度测试：{fold['year']}", flush=True)
        signals = generate_frozen_signals(panel, selected, fold, cfg)
        all_signals.append(signals)
        result = run_portfolio(signals, market, fold["test_start"], fold["test_end"], cfg)
        result["audit"] = audit_result(result, market, cfg)
        out = directory / f"year_{fold['year']}" / "backtest"
        save_result(out, result, selected, cfg, f"XGBoost CTA择时：{fold['year']}年度独立账户")
        annual.append({"year": fold["year"], "coverage": f"{result['metrics']['start']}至{result['metrics']['end']}",
                       "validation_sufficient": selected["validation_sufficient"], "rule_id": selected["rule_id"], **result["metrics"]})
        print(f"年度{fold['year']}：收益{result['metrics']['total_return']:.2%}，夏普{result['metrics']['sharpe']:.3f}", flush=True)
    signals = pd.concat(all_signals, ignore_index=True)
    save_pickle(directory / "frozen_test_signals.pkl", signals)
    result = run_portfolio(signals, market, folds[0]["test_start"], folds[-1]["test_end"], cfg)
    result["audit"] = audit_result(result, market, cfg)
    continuous_rule = {"rule_id": "annual_frozen_switch", "selection": "每年度使用本年度开始前冻结的模型；原持仓沿用入场退出方案"}
    save_result(directory / "continuous", result, continuous_rule, cfg, "XGBoost CTA择时：跨年连续账户")
    stress_cfg = {**cfg, "slippage_rate": cfg["slippage_rate"] * 2}
    stress = run_portfolio(signals, market, folds[0]["test_start"], folds[-1]["test_end"], stress_cfg)
    stress["audit"] = audit_result(stress, market, stress_cfg)
    save_result(directory / "cost_stress", stress, continuous_rule, stress_cfg, "XGBoost CTA择时：双倍滑点压力测试")
    if cfg["fit_deployment_model"]:
        last = selected_rules[-1]
        profile = next(p for p in cfg["model_profiles"] if p["name"] == last["model_id"].split("_", 1)[1])
        deployment_fold = {"year": "deployment", "train_start": pd.Timestamp(cfg["start_date"]),
                           "train_end": pd.Timestamp(cfg["end_date"]) + pd.Timedelta(days=1)}
        fit_model(panel, labels, deployment_fold, last["exit_profile"], profile, cfg, directory / "deployment_candidate")
        write_json(directory / "deployment_candidate" / "status.json", {"independent_test": False,
            "live_approved": False, "reason": "包含2026的全量候选，预测分布改变，必须重新进行未来校准和独立模拟盘验证",
            "research_rule_reference": last})
    for name, digest in frozen.items():
        if calculate_sha256(directory / name) != digest:
            raise AssertionError(f"测试后模型或冻结规则发生变化：{name}")
    current_sources = {name: calculate_sha256(root / name) for name in sources}
    if current_sources != sources:
        write_json(directory / "source_change_notice.json", {"before": sources, "after": current_sources})
        raise AssertionError("运行过程中源码变动，需审计后才能接受报告")
    save_summary(directory, pd.DataFrame(annual), result, stress, manifest, cfg, folds)
    return directory
