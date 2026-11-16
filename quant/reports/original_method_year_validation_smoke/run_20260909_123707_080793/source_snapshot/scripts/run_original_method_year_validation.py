"""在quant根目录运行原XGBoost/CTA年度隔离重训和双执行回顾，输入YAML及旧156列定义。

train拟合/校准/内部验证后冻结；audit重建真实核心并重载；evaluate仅一次四年双方法双口径。
新增模型、pkl、中文报告与逐笔文件，不覆盖历史目录，不连接券商。未来年份训练仅属回顾。
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys

import joblib
import numpy as np
import pandas as pd
import yaml
from xgboost import XGBRegressor, Booster, DMatrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quant_backtest.original_method_data import load_original_panel, calculate_file_hash
from quant_backtest.original_method_portfolio import run_original_portfolio
from quant_backtest.annual_panel import write_json
from quant_backtest.annual_portfolio import MarketStore
from quant_backtest.learned_exit_models import build_research_folds, select_window_rows
from scripts.run_multi_factor_cta_backtests import add_cta_factor_scores
from scripts.run_annual_method_validation import save_result
from scripts.audit_family_path_results import audit_account_frames


METHODS = {"method1": "原XGBoost", "method2": "原XGBoost加CTA"}
MODES = {"legacy": "旧式成交有偏对照", "constrained": "信号后成交及容量约束"}
SOURCES = ["quant_backtest/original_method_data.py", "quant_backtest/original_method_portfolio.py",
    "scripts/run_original_method_year_validation.py", "quant_backtest/annual_panel.py", "quant_backtest/annual_portfolio.py",
    "quant_backtest/factor_library.py", "quant_backtest/bounded_training.py", "quant_backtest/learned_exit_models.py",
    "scripts/run_weekly_high_vol_lot_exit_backtests.py", "scripts/run_multi_factor_cta_backtests.py",
    "scripts/run_all_a_method_backtests.py", "scripts/run_annual_method_validation.py", "scripts/audit_family_path_results.py"]
LIMITATIONS = """- 本轮是原XGBoost/CTA策略的年度回顾，不是最近的LightGBM/净收益模型族。前三折含未来年份训练；2026仅1至4月，不能当当年可部署或全新未见样本。
- 固定156列来自历史因子筛选，历史数据已经反复查看；重新按折拟合不消除历史特征选择及研究选择偏差。
- 保留原24个观测分钟收益截面排名目标、XGBoost参数、CTA公式和ATR240的2/4倍退出；非原二进制模型逐年自测。门槛按每折内部校准top1分数90%分位，不复制旧模型绝对分值。
- 每个非留出年按60%/20%/20%交易日切核心/校准/验证，片段起点隔离7自然日，模型只拟合核心，不在校准后重训。验证不选择新参数，不按其成绩删除年度或停用原策略。
- 为隔离年份，递推/累计因子每年初始化，有限因子仅在该年观察241条后复用；训练和校准不使用跨年周池。全部截面贡献者来源同年，核心收益排名在剔除跨窗口标签后重新计算。
- 沿用已有周度高波动300股缓存和24交易分钟公共时钟，修正原逐股票全历史行号采样及按未来缺标签删推断股票；因此不是原2026年7月报告数值的逐字复现。当前名称ST/退过滤、幸存者及公司行动/复权缺失偏差仍在。
- 旧式口径直接调用原lot函数，卖出读取成交分钟价格/高点/ATR并同刻成交，属于有偏对照；其预算取成交时卖出后权益，零滑点/无限容量/固定卖出万五税率，不能作为实盘夏普验收。
- 约束版只读已完成信号分钟，精确下一分钟尝试成交；入场ATR取信号值，保留2/4原公式（ATR变大时线可放松），不追加近期研究的20/40倍、3/5日期限或离池退出。
- 两口径均前5、每批最多总资产10%、隔日可独立加仓、T+1、100股、万八免5。约束版追加双边万五滑点、分钟5%共享成交额和前收±9.5%阻成交代理，税率按日期；不等于完整涨跌停排队模型。
- 约束版退出遇T+1/缺价/容量不足会保留待退出或部分卖出；旧式函数不持久化受阻止损意图。两个口径的差异不只滑点，不应全部归因手续费。
- 标准曲线由全部公共日终估值计算日频夏普（含现金日、ddof=1、252日、无风险0），年化收益按实际自然日365.25。旧式原bar指标单独留档，不能与标准日频夏普混用。
- 各年初100万元、年末按市值保留持仓，不虚构强平；两个方法共享同折XGBoost模型，不是两个独立风险来源。不根据外层成绩回选方法/口径/年度。
"""


def load_config(path: Path, smoke: bool, workers: int | None) -> dict:
    """读取path YAML并解析根目录路径；smoke只缩小8股/树数/截面，workers是一次线程覆盖。"""
    cfg = yaml.safe_load(path.read_text())
    for name in ["data_root", "annual_cache", "original_model_run", "output_dir", "cache_dir"]:
        cfg[name] = str((ROOT / cfg[name]).resolve())
    cfg["is_smoke"] = smoke
    if workers:
        cfg["workers"] = workers
    if smoke:
        cfg.update(max_symbols=8, min_cross_section=1, output_dir=cfg["output_dir"] + "_smoke")
        cfg["model"] = {**cfg["model"], "n_estimators": 5}
    if cfg["entry_top_n"] != 5 or cfg["entry_equity_fraction"] != .1 or cfg["lot_size"] != 100:
        raise ValueError("必须保留确认的前5/总资产10%/100股")
    if cfg["gate_quantile"] != .9 or cfg["cta_blend"] != .0005 or cfg["sample_minutes"] != 24:
        raise ValueError("必须保留原模型90%门槛、CTA幅度和24分钟采样")
    cfg["report_limitations"] = LIMITATIONS
    return cfg


def select_original_rows(panel: pd.DataFrame, windows: list, cfg: dict, fit: bool) -> pd.DataFrame:
    """按windows选panel，来源和周池必须同年；fit剔除跨边界标签后重新排名，cfg控制截面数。"""
    known = (panel.source_start.dt.year == panel.datetime.dt.year) & (panel.pool_source_start.dt.year == panel.datetime.dt.year)
    frame = select_window_rows(panel.loc[known], windows, require_label=fit)
    if fit:
        frame = frame.loc[np.isfinite(frame.forward_return)].copy()
        frame["target_rank"] = frame.groupby("datetime").forward_return.rank(pct=True).astype("float32")
        count = frame.groupby("datetime").symbol.transform("size")
    else:
        count = frame.cross_section_count
    return frame.loc[count >= cfg["min_cross_section"]].sort_values(["datetime", "symbol"]).reset_index(drop=True)


def hash_training_frame(frame: pd.DataFrame) -> str:
    """对frame真实输入/列序/类型生成指纹，不把DataFrame索引编号误作样本身份。"""
    meta = json.dumps([(str(c), str(frame[c].dtype)) for c in frame], ensure_ascii=False).encode()
    return hashlib.sha256(meta + pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes()).hexdigest()


def generate_original_scores(model, frame: pd.DataFrame, features: list, cfg: dict) -> pd.DataFrame:
    """用model对frame固定features预测原截面排名，cfg只给原CTA幅度；输出严格不含未来标签。"""
    out = frame[["datetime", "symbol", *[c for c in frame if c.startswith("cs_rank_")]]].copy()
    if out.empty:
        return pd.DataFrame(columns=["datetime", "symbol", "prediction", "cta_trend_breakout_score"])
    out["prediction"] = model.predict(out[features]).astype("float32")
    return add_cta_factor_scores(out, cfg["cta_blend"])[["datetime", "symbol", "prediction", "cta_trend_breakout_score"]]


def calculate_original_gates(scores: pd.DataFrame, cfg: dict) -> dict:
    """从scores内部校准分数按cfg的固定top1分位分别取两门槛，不读未来可成交性或收益。"""
    if scores.empty:
        raise ValueError("内部校准没有可用截面，不得用测试数据补门槛")
    result = {}
    for method in METHODS:
        column = "prediction" if method == "method1" else "cta_trend_breakout_score"
        gate = float(scores.groupby("datetime")[column].max().quantile(cfg["gate_quantile"]))
        if not np.isfinite(gate):
            raise ValueError("门槛不是有限数")
        result[method] = gate
    return result


def freeze_run(directory: Path) -> None:
    """冻结directory当前全部训练产物与源码，之后只有根final_report允许外层后更新。"""
    files = {str(p.relative_to(directory)): calculate_file_hash(p) for p in directory.rglob("*") if p.is_file()}
    write_json(directory / "frozen_manifest.json", {"frozen_at": datetime.now().isoformat(), "files": files})


def load_frozen_run(directory: Path) -> tuple:
    """读取directory前核验全部冻结文件/源码/软件，不绕过旧快照指纹，返回cfg/folds/data。"""
    frozen = json.loads((directory / "frozen_manifest.json").read_text())
    for name, value in frozen["files"].items():
        if name == "final_report.md" and (directory / "test_started.json").exists():
            continue
        if calculate_file_hash(directory / name) != value:
            raise ValueError(f"冻结文件改变：{name}")
    protocol = json.loads((directory / "research_protocol.json").read_text())
    for name, value in protocol["source_hashes"].items():
        if calculate_file_hash(ROOT / name) != value:
            raise ValueError(f"源码已改变，需对应快照：{name}")
    for name, value in protocol["packages"].items():
        if importlib.metadata.version(name) != value:
            raise ValueError(f"软件版本已变：{name}")
    cfg = yaml.safe_load((directory / "resolved_config.yaml").read_text())
    panel, raw, manifest = load_original_panel(cfg)
    if manifest != json.loads((directory / "data_metadata.json").read_text()):
        raise ValueError("缓存实际内容与训练时不同")
    market = MarketStore(raw)
    folds = build_research_folds(market.times, cfg)
    expected = json.loads((directory / "folds.json").read_text())
    if json.loads(json.dumps(folds, default=str)) != expected:
        raise ValueError("实际年度切分已改变")
    return cfg, folds, panel, market, manifest


def write_output_guide(directory: Path) -> None:
    """在directory说明训练、模型、逐年双口径和独立审计文件的角色，不宣称已有盈利。"""
    text = """# 输出文件说明

| 文件 | 用途 |
|---|---|
| resolved_config.yaml / research_protocol.json | 实际参数、原策略身份、两口径、来源和软件版本；先于训练保存 |
| data_metadata.json / folds.json | 特征pkl内容哈希、固定156列及四折边界 |
| source_snapshot/ | 本轮及共用依赖代码快照，非独立安装包 |
| 年份/xgboost_model.joblib / xgboost_model.json | 该折新训XGBoost，两个策略共享，不是旧全量模型 |
| 年份/model_metadata.json / feature_list.txt / feature_manifest.csv | 原列序/中文含义、真实核心输入与标签指纹、参数及来源年份 |
| 年份/frozen_rules.json | 两个独立校准门槛，前5/10%和原ATR2/4规则 |
| 年份/validation_rule_metrics.csv | 三个内部验证片段的两方法两口径表现，仅诊断，不回选 |
| training_summary.csv / training_report.md | 核心样本/门槛和测试前冻结摘要 |
| frozen_manifest.json / training_audit.json | 文件冻结及真实核心/无标签推断/原生模型重载审计 |
| test_started.json | 单次外层启动标记，存在即禁止重跑 |
| test_year_metrics.csv / test_summary.csv | 四年、两方法、两口径各自指标和完整均夏普，不逐年择优 |
| 年份/口径/方法/final_report.md | 单年中文策略报告与限制 |
| 年份/口径/方法/测试集收益曲线.png / test_return_curve.png | 同图双文件名，含收益、回撤和指标 |
| 年份/口径/方法/test_equity_curve.csv | 全部公共时钟净值、现金和暴露，含日终补估值 |
| 年份/口径/方法/test_trade_events.csv | 买卖时刻、信号、独立批次、股数、价格、费用和触发原因 |
| 年份/口径/方法/test_round_trips.csv | 按批次买卖配对，部分卖出单独行 |
| 年份/口径/方法/test_open_positions.csv / test_holdings.csv | 期末未平仓与每日收盘持仓 |
| 年份/口径/方法/test_metrics.csv / strategy_metadata.json | 标准日频指标及冻结规则、受阻计数 |
| 年份/legacy/方法/legacy_native_metrics.json | 旧函数原不规则bar年化，非本轮正式日频比较口径 |
| result_account_audit.csv | 四年两口径账务/约束核对；旧式同刻退出单独标注不通过因果要求 |

legacy表示旧式有偏成交对照，constrained表示先信号后成交与滑点/容量约束。
method1为原XGBoost截面排名，method2为同模型加原CTA微小叠加；不是LightGBM模型族。
两口径均复用同折同一模型和同一门槛，不能看外层后切换再宣称通过。
"""
    (directory / "输出文件说明.md").write_text(text, encoding="utf-8")


def run_training(cfg: dict) -> Path:
    """按cfg预写协议，复用pkl、拟合四模型、分别校准门槛，验证只诊断后冻结；不看外层。"""
    directory = Path(cfg["output_dir"]) / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    source_hashes = {}
    for name in SOURCES:
        target = directory / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
        source_hashes[name] = calculate_file_hash(target)
    write_json(directory / "research_protocol.json", {"created": datetime.now().isoformat(), "source_hashes": source_hashes,
        "packages": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "xgboost", "TA-Lib", "scikit-learn"]},
        "identity": "原156列XGB截面rank，原CTA公式，ATR240初始2/跟踪4，不换净收益模型或期限退出",
        "selection": "固定原参数/退出和90%分位校准；内部验证只诊断，不按成绩停用或切换",
        "modes": MODES, "limitations": LIMITATIONS,
        "criterion": "分别完整四年日频夏普等权平均；旧式有同刻退出偏差，不能作为实盘目标验收"})
    print(f"原XGBoost/CTA年度重训：{directory}", flush=True)
    panel, raw, manifest = load_original_panel(cfg)
    market = MarketStore(raw)
    folds = build_research_folds(market.times, cfg)
    write_json(directory / "data_metadata.json", manifest)
    write_json(directory / "folds.json", folds)
    summaries = []
    features = manifest["features"]
    for fold in folds:
        path = directory / str(fold["year"])
        path.mkdir()
        core = select_original_rows(panel, fold["fit"], cfg, True)
        if len(core) < 100:
            raise ValueError(f"{fold['year']}实际核心不足100，不能用测试行补训练")
        model = XGBRegressor(**cfg["model"], objective="reg:squarederror", tree_method="hist", n_jobs=cfg["workers"], eval_metric="rmse")
        print(f"  {fold['year']}原XGBoost拟合：{len(core)}行，156列", flush=True)
        model.fit(core[features], core.target_rank)
        joblib.dump(model, path / "xgboost_model.joblib")
        model.save_model(path / "xgboost_model.json")
        (path / "feature_list.txt").write_text("\n".join(features) + "\n")
        from quant_backtest.factor_library import infer_factor_metadata
        descriptions = infer_factor_metadata([c.removeprefix("cs_rank_") for c in features])
        descriptions.insert(0, "model_feature", features)
        descriptions.to_csv(path / "feature_manifest.csv", index=False, encoding="utf-8-sig")
        hashed_columns = ["datetime", "symbol", "source_start", "pool_source_start", "label_end", "target_rank", *features]
        audit = {"year": fold["year"], "features": features, "params": cfg["model"], "rows": len(core),
            "core_sha256": hash_training_frame(core[hashed_columns]), "columns_hashed": hashed_columns,
            "years": {str(y): int(n) for y, n in core.groupby(core.datetime.dt.year).size().items()},
            "source_min": str(core.source_start.min()), "label_max": str(core.label_end.max()), "objective": "24分钟未来收益截面百分位排名"}
        write_json(path / "model_metadata.json", audit)
        calibration = select_original_rows(panel, fold["calibration"], cfg, False)
        gates = calculate_original_gates(generate_original_scores(model, calibration, features, cfg), cfg)
        write_json(path / "frozen_rules.json", {"thresholds": gates, "gate_quantile": cfg["gate_quantile"],
            "entry_top_n": 5, "entry_equity_fraction": .1, "initial": 2., "trailing": 4., "atr_period": 240, "max_days": 0})
        validation = generate_original_scores(model, select_original_rows(panel, fold["validation"], cfg, False), features, cfg)
        metrics = []
        for mode in MODES:
            for method in METHODS:
                for window in fold["validation"]:
                    result = run_original_portfolio(validation, market, window["start"], window["end"], method, gates[method], mode, cfg)
                    metrics.append({"mode": mode, "method": method, "validation_year": window["year"], **result["metrics"]})
        pd.DataFrame(metrics).to_csv(path / "validation_rule_metrics.csv", index=False, encoding="utf-8-sig")
        summaries.append({"year": fold["year"], "core_rows": len(core), "core_2026": audit["years"].get("2026", 0), **gates})
        print(f"  {fold['year']}两方法、两口径内部验证完成，未读取外层", flush=True)
    pd.DataFrame(summaries).to_csv(directory / "training_summary.csv", index=False, encoding="utf-8-sig")
    text = "# 原XGBoost与CTA：训练冻结\n\n尚无本轮外层成绩，不承诺夏普超过3。\n\n"
    text += pd.DataFrame(summaries).rename(columns={"year": "留出年", "core_rows": "核心行数", "core_2026": "2026实际核心行", "method1": "XGB门槛", "method2": "CTA门槛"}).to_markdown(index=False)
    for name in ["training_report.md", "final_report.md"]:
        (directory / name).write_text(text + "\n\n## 限制\n\n" + LIMITATIONS, encoding="utf-8")
    write_output_guide(directory)
    freeze_run(directory)
    print(f"四个原XGBoost模型及八门槛已冻结：{directory}", flush=True)
    return directory


def run_audit(directory: Path) -> None:
    """重建directory四个模型的实际核心及校准门槛，并核对原生JSON预测；只读内部数据。"""
    cfg, folds, panel, _, manifest = load_frozen_run(directory)
    checks = []
    for fold in folds:
        path = directory / str(fold["year"])
        meta = json.loads((path / "model_metadata.json").read_text())
        core = select_original_rows(panel, fold["fit"], cfg, True)
        if fold["year"] in core.datetime.dt.year.unique() or hash_training_frame(core[meta["columns_hashed"]]) != meta["core_sha256"]:
            raise ValueError("核心输入或留出年隔离失败")
        if len(core) != meta["rows"] or manifest["features"] != meta["features"]:
            raise ValueError("模型列序/样本数不同")
        model = joblib.load(path / "xgboost_model.joblib")
        if any(model.get_params()[k] != v for k, v in cfg["model"].items()):
            raise ValueError("不是原参数模型")
        booster = Booster()
        booster.load_model(path / "xgboost_model.json")
        sample = core[meta["features"]].iloc[:100]
        np.testing.assert_allclose(model.predict(sample), booster.predict(DMatrix(sample)), rtol=0, atol=1e-7)
        calibration = select_original_rows(panel, fold["calibration"], cfg, False)
        scores = generate_original_scores(model, calibration, meta["features"], cfg)
        rules = json.loads((path / "frozen_rules.json").read_text())
        if calculate_original_gates(scores, cfg) != rules["thresholds"]:
            raise ValueError("两方法校准门槛不一致")
        poisoned = calibration.drop(columns=["target_rank", "forward_return", "label_end"])
        pd.testing.assert_frame_equal(scores, generate_original_scores(model, poisoned, meta["features"], cfg))
        checks.append({"year": fold["year"], "rows": len(core), "passed": True, "core_2026": meta["years"].get("2026", 0)})
    write_json(directory / "training_audit.json", {"passed": True, "models": checks,
        "frozen_manifest_sha256": calculate_file_hash(directory / "frozen_manifest.json"), "created": datetime.now().isoformat()})
    print("四折原XGBoost真实核心、校准与原生重载审计通过", flush=True)


def audit_original_result(result: dict, mode: str, cfg: dict) -> dict:
    """重算result账务；mode旧式允许已披露同刻退出/旧预算，约束式用独立完整CSV审计。"""
    frames = {k: result[k] for k in ["curve", "events", "trips", "open", "holdings"]}
    frames["metrics"] = pd.DataFrame([result["metrics"]])
    if mode == "constrained":
        return audit_account_frames(frames, cfg)
    events, curve, trips, opened = [result[k] for k in ["events", "curve", "trips", "open"]]
    buys = events.loc[events.side == "buy"]
    assert ((buys.datetime - buys.signal_datetime) == pd.Timedelta(minutes=1)).all()
    assert (trips.sell_datetime.dt.normalize() > trips.buy_datetime.dt.normalize()).all()
    assert (events.shares % 100 == 0).all() and (curve.cash >= -1e-6).all() and curve.exposure.between(0, 1 + 1e-8).all()
    assert (buys.notional <= buys.signal_equity * .1 + 1e-5).all()
    fees = events.notional * (cfg["commission_rate"] + cfg["transfer_rate"] + np.where(events.side == "sell", cfg["stamp_rate_from_20230828"], 0))
    np.testing.assert_allclose(events.fee.to_numpy(dtype=float), fees.to_numpy(dtype=float), rtol=0, atol=1e-7)
    flow = np.where(events.side == "buy", -events.notional - events.fee, events.notional - events.fee)
    changes = pd.Series(np.asarray(flow, dtype=float), index=events.datetime).groupby(level=0).sum()
    expected_cash = cfg["initial_cash"] + changes.reindex(curve.datetime, fill_value=0).cumsum().to_numpy()
    np.testing.assert_allclose(curve.cash, expected_cash, atol=1e-5, rtol=0)
    pnl = trips.net_pnl.sum() + (opened.shares * (opened.mark_price - opened.buy_price) - opened.buy_fee).sum()
    np.testing.assert_allclose(curve.equity.iloc[-1] - cfg["initial_cash"], pnl, atol=1e-5, rtol=0)
    daily = curve.groupby(curve.datetime.dt.normalize()).equity.last()
    returns = daily.pct_change(fill_method=None)
    returns.iloc[0] = daily.iloc[0] / cfg["initial_cash"] - 1
    std = returns.std(ddof=1)
    sharpe = float(returns.mean() / std * np.sqrt(252)) if std > 0 else 0.
    np.testing.assert_allclose(result["metrics"]["sharpe"], sharpe, atol=1e-8, rtol=0)
    return {"passed": True, "cash_error": float(np.max(np.abs(curve.cash - expected_cash))),
            "pnl_error": float(abs(curve.equity.iloc[-1] - cfg["initial_cash"] - pnl)), "sharpe": sharpe,
            "same_bar_exit_biased": True, "budget_basis": "成交时卖出后权益，不是先验信号权益"}


def run_evaluation(directory: Path) -> None:
    """对directory全部冻结折一次外层，固定四组策略不择优，保存标准图表/逐笔/账务核对。"""
    if (directory / "test_started.json").exists():
        raise ValueError("外层已经启动，禁止重复测试")
    cfg, folds, panel, market, manifest = load_frozen_run(directory)
    audit = json.loads((directory / "training_audit.json").read_text())
    if not audit["passed"] or audit["frozen_manifest_sha256"] != calculate_file_hash(directory / "frozen_manifest.json"):
        raise ValueError("训练审计未通过或不对应本轮冻结")
    write_json(directory / "test_started.json", {"started": datetime.now().isoformat(), "is_smoke": cfg["is_smoke"]})
    metrics, checks = [], []
    for fold in folds:
        path = directory / str(fold["year"])
        model = joblib.load(path / "xgboost_model.joblib")
        test = panel.loc[(panel.datetime >= fold["test_start"]) & (panel.datetime < fold["test_end"])
                         & (panel.cross_section_count >= cfg["min_cross_section"])]
        scores = generate_original_scores(model, test, manifest["features"], cfg)
        gates = json.loads((path / "frozen_rules.json").read_text())["thresholds"]
        for mode in MODES:
            for method in METHODS:
                result = run_original_portfolio(scores, market, fold["test_start"], fold["test_end"], method, gates[method], mode, cfg)
                checks.append({"year": fold["year"], "mode": mode, "method": method, **audit_original_result(result, mode, cfg)})
                result.update(report_note="原策略留一年回顾；前三折含未来训练，旧式成交有偏，2026仅1至4月。", limitations=LIMITATIONS)
                rule = {"rule_id": "atr_initial2_trail4_atr240", "threshold": gates[method],
                        "selection": "原规则固定；本折内部90%门槛校准；验证只诊断，不按测试选择", "mode": mode}
                target = path / mode / method
                save_result(target, result, f"{fold['year']} {'烟测 ' if cfg['is_smoke'] else ''}{METHODS[method]}：{MODES[mode]}", rule, cfg)
                if mode == "legacy":
                    write_json(target / "legacy_native_metrics.json", result["legacy_native_metrics"])
                    guide = target / "输出文件说明.md"
                    guide.write_text(guide.read_text() + "\n旧式signal_equity为成交时卖出后预算基数；卖出signal_datetime等于成交时间，明确存在同刻退出偏差。legacy_native_metrics.json为旧bar指标，正式比较使用日终重估后的test_metrics.csv。\n", encoding="utf-8")
                metrics.append({"year": fold["year"], "mode": mode, "method": method, **result["metrics"]})
                print(f"外层完成：{fold['year']}/{mode}/{method}", flush=True)
    table = pd.DataFrame(metrics)
    summaries = []
    for mode in MODES:
        for method in METHODS:
            selected = table.loc[(table["mode"] == mode) & (table.method == method)]
            valid = len(selected) == 4 and (selected.annual_volatility > 0).all()
            mean = float(selected.sharpe.mean()) if valid else None
            summaries.append({"mode": mode, "method": method, "valid_years": int((selected.annual_volatility > 0).sum()),
                "mean_sharpe": mean, "numeric_above3": bool(valid and mean > cfg["sharpe_target"]),
                "eligible_goal_evidence": bool(mode == "constrained" and not cfg["is_smoke"]), "live_ready": False})
    table.to_csv(directory / "test_year_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summaries).to_csv(directory / "test_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(checks).to_csv(directory / "result_account_audit.csv", index=False, encoding="utf-8-sig")
    display = table[["year", "mode", "method", "total_return", "sharpe", "max_drawdown", "buy_count"]].copy()
    display["mode"] = display["mode"].map(MODES)
    display["method"] = display.method.map(METHODS)
    display = display.rename(columns={"year": "留出年", "mode": "执行口径", "method": "方法", "total_return": "收益率", "sharpe": "夏普", "max_drawdown": "最大回撤", "buy_count": "买入次数"})
    text = "# 原XGBoost与CTA：四年双执行回顾\n\n" + display.to_markdown(index=False) + "\n\n"
    for item in summaries:
        text += f"- {METHODS[item['method']]}，{MODES[item['mode']]}：完整四年均夏普{item['mean_sharpe']}，数值超过3={item['numeric_above3']}。\n"
    if cfg["is_smoke"]:
        text += "\n仅8股流程烟测，不作为盈利证明。\n"
    text += "\n## 年度图表和逐笔记录\n\n"
    for fold in folds:
        for mode in MODES:
            for method in METHODS:
                text += f"- [{fold['year']} {METHODS[method]} {MODES[mode]}]({fold['year']}/{mode}/{method}/final_report.md)\n"
    (directory / "final_report.md").write_text(text + "\n## 限制\n\n" + LIMITATIONS, encoding="utf-8")
    print(f"原两策略四年双执行回顾全部完成：{directory}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/original_method_year_validation.yaml")
    parser.add_argument("--phase", choices=["build", "train", "audit", "evaluate"], default="train")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    if args.phase in {"audit", "evaluate"}:
        if args.run_dir is None:
            parser.error("audit/evaluate必须给--run-dir")
        (run_audit if args.phase == "audit" else run_evaluation)(args.run_dir.resolve())
    else:
        cfg = load_config(args.config, args.smoke, args.workers)
        if args.phase == "build":
            _, _, metadata = load_original_panel(cfg)
            print({k: v for k, v in metadata.items() if k not in {"features", "columns"}})
        else:
            run_training(cfg)
