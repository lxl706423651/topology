"""固定底仓做T的年度开仓模型：训练内选特征、留一年划分和独立推断。

输入为 annual_panel 的共同时钟截面秩面板及显式配置；只写调用者指定的模型
目录，不修改原始行情、已有模型或缓存。模型学习次交易日相同时刻收益的截面
排名；这与旧策略24分钟标签不同。CTA沿用旧技术因子微调和top5开仓结构。
开仓门槛由外层训练范围的内层留出预测校准；最终模型重训后保持数值门槛固定。
留一年训练包含未来年份时仅是回顾性检验，不等同可部署的历史前向回测。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
import xgboost
from xgboost import XGBRegressor

from scripts.run_multi_factor_cta_backtests import add_cta_factor_scores


def build_year_folds(panel: pd.DataFrame, years: Iterable[int], embargo_days: int = 7) -> list[dict]:
    """按panel实际年份构造years的外层留一年折；最新训练年留作内层验证。

    返回训练/验证年份和前向或回顾性协议。这里只划分年份，标签清除与年度后
    embargo由select_model_training_rows执行；至少需要三个实际数据年份。
    """
    available = sorted(pd.to_datetime(panel["datetime"]).dt.year.unique().astype(int).tolist())
    if len(available) < 3:
        raise ValueError("年度嵌套验证至少需要三个有行情的年份")
    if embargo_days < 0:
        raise ValueError("embargo_days不能为负数")
    folds = []
    for year in sorted(set(map(int, years))):
        if year not in available:
            raise ValueError(f"验证年份{year}不在实际面板年份{available}内")
        training = [candidate for candidate in available if candidate != year]
        inner = training[-1]
        folds.append({"test_year": year, "train_years": training,
                      "inner_validation_year": inner,
                      "core_train_years": [candidate for candidate in training if candidate != inner],
                      "protocol": "forward" if max(training) < year else "leave_one_year_out_retrospective",
                      "embargo_days": int(embargo_days)})
    return folds


def select_model_training_rows(panel: pd.DataFrame, train_years: Iterable[int], cfg: dict,
                               excluded_years: Iterable[int] = ()) -> pd.DataFrame:
    """从panel选出train_years内标签完整的训练行，清除跨留出年标签及年度后隔离期。

    需要datetime、label_end、forward_return、target_rank列。仅训练过滤未来标签；
    推断函数不会因未来价格/标签缺失删除股票。cfg的embargo_days默认7个日历日。
    """
    training = sorted(set(map(int, train_years)))
    excluded = sorted(set(map(int, excluded_years)))
    if not training or set(training).intersection(excluded):
        raise ValueError("训练年份须非空且与排除年份互斥")
    timestamp = pd.to_datetime(panel["datetime"])
    label_end = pd.to_datetime(panel["label_end"])
    mask = timestamp.dt.year.isin(training) & label_end.dt.year.isin(training)
    mask &= label_end > timestamp
    mask &= np.isfinite(pd.to_numeric(panel["forward_return"], errors="coerce"))
    mask &= np.isfinite(pd.to_numeric(panel["target_rank"], errors="coerce"))
    min_assets = int(cfg.get("min_cross_section", 30))
    counts = panel.get("cross_section_count")
    if counts is None:
        counts = panel.groupby("datetime")["symbol"].transform("size")
    mask &= counts >= min_assets
    embargo_days = int(cfg.get("embargo_days", 7))
    if embargo_days < 0:
        raise ValueError("embargo_days不能为负数")
    for year in excluded:
        first_after = pd.Timestamp(year=year + 1, month=1, day=1)
        mask &= ~timestamp.between(first_after, first_after + pd.Timedelta(days=embargo_days), inclusive="left")
    result = panel.loc[mask]
    if result.empty:
        raise ValueError(f"训练年份{training}清除标签与隔离期后没有可训练样本")
    return result


def select_training_features(training: pd.DataFrame, cfg: dict) -> tuple[list[str], pd.DataFrame, dict]:
    """仅用training的秩因子及target_rank选特征；返回确定性特征清单、审计表和统计。

    cfg控制覆盖率、最大相关系数、特征上限及抽样量；相关性和RankIC在固定种子
    的训练子样本计算，缺失交给XGBoost处理，不用验证或测试标签选择特征。
    """
    columns = sorted(column for column in training.columns if column.startswith("cs_rank_"))
    if not columns:
        raise ValueError("训练面板缺少cs_rank_因子列")
    max_features = int(cfg.get("max_features", 154))
    sample_rows = int(cfg.get("feature_selection_sample_rows", 20000))
    min_coverage = float(cfg.get("feature_min_coverage", 0.80))
    max_correlation = float(cfg.get("feature_max_correlation", 0.90))
    if max_features < 1 or sample_rows < 2 or not 0 <= min_coverage <= 1 or not 0 < max_correlation <= 1:
        raise ValueError("特征上限、抽样量、覆盖率或相关系数配置非法")
    seed = int(cfg.get("model", {}).get("random_state", cfg.get("random_state", 42)))
    sample = training.sample(n=min(len(training), sample_rows), random_state=seed)
    sample_features = sample[columns].replace([np.inf, -np.inf], np.nan)
    # 覆盖率覆盖全部训练行，相关性只抽样；二者均不读取留出年份。
    coverage = pd.Series({column: float(np.isfinite(training[column].to_numpy(dtype="float32")).mean())
                          for column in columns})
    distinct = sample_features.nunique(dropna=True)
    ranked = sample_features.rank(pct=True)
    target = sample["target_rank"].rank(pct=True)
    relevance = ranked.corrwith(target).fillna(0.0)
    eligible = [column for column in columns if coverage[column] >= min_coverage and distinct[column] > 1]
    ordered = sorted(eligible, key=lambda column: (-abs(float(relevance[column])), column))
    correlations = sample_features[ordered].corr(method="spearman") if ordered else pd.DataFrame()
    selected, rows = [], []
    for column in ordered:
        corr = correlations.loc[column, selected].abs().dropna() if selected else pd.Series(dtype=float)
        strongest = float(corr.max()) if len(corr) else 0.0
        peer = str(corr.idxmax()) if len(corr) else ""
        if strongest > max_correlation:
            reason = "training_correlation_limit"
        elif len(selected) >= max_features:
            reason = "feature_count_limit"
        else:
            reason = "selected"
            selected.append(column)
        rows.append({"feature": column, "coverage": float(coverage[column]),
                     "training_rank_ic": float(relevance[column]), "selected": reason == "selected",
                     "reason": reason, "max_selected_correlation": strongest, "most_correlated_feature": peer})
    for column in columns:
        if column in eligible:
            continue
        rows.append({"feature": column, "coverage": float(coverage[column]),
                     "training_rank_ic": float(relevance[column]), "selected": False,
                     "reason": "low_coverage" if coverage[column] < min_coverage else "constant_in_training_sample",
                     "max_selected_correlation": 0.0, "most_correlated_feature": ""})
    if not selected:
        raise ValueError("训练期覆盖率与相关性筛选后没有可用因子")
    return selected, pd.DataFrame(rows), {"selection_rows": len(sample), "candidate_feature_count": len(columns),
                                         "selected_feature_count": len(selected), "selection_seed": seed,
                                         "max_features": max_features, "feature_min_coverage": min_coverage,
                                         "feature_max_correlation": max_correlation}


def fit_entry_model(panel: pd.DataFrame, train_years: Iterable[int], cfg: dict,
                    output_dir: Path | str, excluded_years: Iterable[int] = ()) -> tuple[XGBRegressor, list[str], dict]:
    """训练年度开仓XGBoost并保存模型/特征/元数据到output_dir；不评估留出集。

    cfg.model提供固定超参数；hist和CPU线程数由本模块固定，训练标签为年度面板
    的次交易日收益排名。可显式max_train_rows限制拟合行数，完整筛选仍仅用训练集。
    """
    training_years = sorted(set(map(int, train_years)))
    exclusions = sorted(set(map(int, excluded_years)))
    training = select_model_training_rows(panel, training_years, cfg, exclusions)
    min_rows = int(cfg.get("min_training_rows", 100))
    if len(training) < min_rows:
        raise ValueError(f"可训练行{len(training)}少于min_training_rows={min_rows}")
    features, manifest, feature_metadata = select_training_features(training, cfg)
    parameters = {"objective": "reg:squarederror", "n_estimators": 180, "max_depth": 4,
                  "learning_rate": 0.04, "min_child_weight": 40, "subsample": 0.85,
                  "colsample_bytree": 0.85, "reg_lambda": 8.0, "reg_alpha": 0.1, "random_state": 42}
    parameters.update(cfg.get("model", {}))
    parameters.update({"tree_method": "hist", "n_jobs": max(1, int(cfg.get("workers", 4)))})
    fitted_rows = training
    max_rows = int(cfg.get("max_train_rows", 0))
    if max_rows < 0:
        raise ValueError("max_train_rows不能为负数")
    if max_rows and len(training) > max_rows:
        fitted_rows = training.sample(n=max_rows, random_state=int(parameters["random_state"])).sort_values(["datetime", "symbol"])
    model = XGBRegressor(**parameters)
    print(f"开仓模型训练年份{training_years}：{len(fitted_rows):,}行/{len(features)}特征；排除{exclusions}", flush=True)
    model.fit(fitted_rows[features].replace([np.inf, -np.inf], np.nan), fitted_rows["target_rank"].astype("float32"))
    signature = pd.util.hash_pandas_object(fitted_rows[["datetime", "symbol", "label_end"]], index=False).to_numpy().tobytes()
    metadata = {"train_years": training_years, "excluded_years": exclusions,
                "eligible_training_rows": len(training), "fitted_training_rows": len(fitted_rows),
                "training_start": str(training.datetime.min()), "training_end": str(training.datetime.max()),
                "label_end_max": str(pd.to_datetime(training.label_end).max()),
                "fitted_sample_sha256": hashlib.sha256(signature).hexdigest(),
                "model_parameters": parameters, "xgboost_version": xgboost.__version__,
                "feature_names": features, **feature_metadata,
                "embargo_days": int(cfg.get("embargo_days", 7)),
                "label_definition": "次交易日相同采样时刻+1分钟价格/本日采样时刻+1分钟价格-1的截面排名；旧模型为24分钟bar标签",
                "feature_selection_scope": "仅本折训练年份的完整标签样本；不复用旧报告已筛选清单",
                "calibration_scope": "调用方在外层训练年份内用核心模型的内层留出预测校准固定分位数；全训练年份重拟合后沿用该数值门槛，不重新使用拟合内预测或测试分布校准",
                "calibration_limitation": "内层年份同时用于门控校准和T规则选择，其收益属于内部选择表现；重拟合可能改变分数分布，外层零开仓须保留而不能再调阈值",
                "embargo_limitation": "隔离期降低相邻年滚动特征重叠，不重置EMA等长记忆指标；含未来训练年份的留一年折仅为回顾性检验",
                "signal_lag_minutes": 1, "missing_feature_policy": "NaN交给XGBoost原生缺失分支"}
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, directory / "model.joblib")
    model.save_model(directory / "model.json")
    manifest.to_csv(directory / "feature_selection_audit.csv", index=False)
    selected_manifest = pd.DataFrame({"feature": features, "selection_order": range(1, len(features) + 1)})
    selected_manifest = selected_manifest.merge(manifest, on="feature", how="left", validate="one_to_one")
    selected_manifest.to_csv(directory / "used_feature_manifest.csv", index=False)
    (directory / "feature_list.txt").write_text("\n".join(features) + "\n", encoding="utf-8")
    (directory / "model_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return model, features, metadata


def predict_entry_scores(model: XGBRegressor, features: list[str], panel: pd.DataFrame,
                         years: Iterable[int], cfg: dict) -> pd.DataFrame:
    """对years内全部当时可推断股票预测，返回datetime/symbol/prediction/score及截面数。

    不读取forward_return、target_rank、label_end，也不以未来标签是否缺失筛选推断。
    entry_method为xgboost或cta_trend；CTA固定使用旧技术因子偏移，不另训练模型。
    """
    requested = set(map(int, years))
    missing = set(features).difference(panel.columns)
    if missing:
        raise ValueError(f"推断面板缺少训练特征：{sorted(missing)[:10]}")
    mask = pd.to_datetime(panel["datetime"]).dt.year.isin(requested)
    source = panel.loc[mask]
    columns = ["datetime", "symbol", "prediction", "score", "cross_section_count"]
    if source.empty:
        return pd.DataFrame(columns=columns)
    prediction = model.predict(source[features].replace([np.inf, -np.inf], np.nan)).astype("float32")
    output = source[["datetime", "symbol"]].copy()
    output["prediction"] = prediction
    method = cfg.get("entry_method", "cta_trend")
    if method == "xgboost":
        output["score"] = prediction
    elif method in {"cta_trend", "cta_trend_breakout"}:
        # CTA可使用未被XGBoost选中的固定风格因子；其权重是事先给定的，不做标签筛选。
        factor_columns = [column for column in source if column.startswith("cs_rank_")]
        cta_input = source[factor_columns].copy()
        cta_input["prediction"] = prediction
        output["score"] = add_cta_factor_scores(cta_input, float(cfg.get("cta_blend", 0.0005)))["cta_trend_breakout_score"].to_numpy()
    else:
        raise ValueError(f"未知entry_method：{method}")
    output["cross_section_count"] = source["cross_section_count"].to_numpy() if "cross_section_count" in source else output.groupby("datetime")["symbol"].transform("size")
    return output[columns].sort_values(["datetime", "symbol"]).reset_index(drop=True)


def rank_entry_scores(scores: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """仅凭scores当前截面进行稳定排序；先过滤不足min_cross_section的信号时刻。"""
    counts = scores.groupby("datetime")["symbol"].transform("size")
    valid = counts >= int(cfg.get("min_cross_section", 30))
    valid &= np.isfinite(pd.to_numeric(scores["score"], errors="coerce"))
    return scores.loc[valid].sort_values(["datetime", "score", "symbol"], ascending=[True, False, True])


def calibrate_entry_gate(scores: pd.DataFrame, cfg: dict) -> dict:
    """按调用方提供的外层训练范围预测校准固定gate_quantile的top1分数门槛。

    年度流程传入核心模型的内层留出年预测，避免拟合内过高分数作为开仓门槛；
    内层年同时用于门控校准和T参数选择，最终模型重拟合后冻结此数值门槛。
    最新旧策略只要求单票分数超过门槛，因此spread_quantile默认禁用；只有显式
    配置时才增加top1/top2分差门控。调用方负责确保scores不包含外层测试预测。
    """
    ordered = rank_entry_scores(scores, cfg)
    if ordered.empty:
        raise ValueError("外层训练范围预测没有满足截面数量要求的门控校准样本")
    quantile = float(cfg.get("gate_quantile", 0.90))
    if not 0 <= quantile <= 1:
        raise ValueError("gate_quantile须在[0,1]内")
    first = ordered.groupby("datetime", sort=False).head(1).set_index("datetime")["score"]
    threshold = float(first.quantile(quantile))
    spread_quantile = cfg.get("spread_quantile")
    spread_threshold = None
    if spread_quantile is not None:
        spread_quantile = float(spread_quantile)
        if not 0 <= spread_quantile <= 1:
            raise ValueError("spread_quantile须为null或[0,1]内分位数")
        second = ordered.loc[ordered.groupby("datetime").cumcount() == 1].set_index("datetime")["score"]
        spreads = (first - second).dropna()
        if spreads.empty:
            raise ValueError("分差门控至少需要每个采样点两只股票")
        spread_threshold = float(spreads.quantile(spread_quantile))
    return {"threshold": threshold, "gate_quantile": quantile, "spread_quantile": spread_quantile,
            "spread_threshold": spread_threshold, "calibration_periods": len(first),
            "calibration_start": str(first.index.min()), "calibration_end": str(first.index.max()),
            "calibration_scope": "调用者提供的外层训练范围预测；年度流程采用核心模型内层留出预测，固定分位数，最终重拟合后冻结数值门槛，不读取外层测试分布或收益"}


def select_entry_candidates(scores: pd.DataFrame, gate: dict, cfg: dict) -> pd.DataFrame:
    """按gate固定门槛取每个时刻top_n，保留scores原有列；不读取未来成交或标签。

    与最新top5旧规则一致：每只候选均须超过门槛，跌出候选列表不代表平仓。
    可成交性、每日同股新开仓限制与10%资产 sizing 由底仓执行引擎判断。
    """
    ordered = rank_entry_scores(scores, cfg)
    top_n = int(cfg.get("entry_top_n", 5))
    if top_n < 1:
        raise ValueError("entry_top_n须大于0")
    if ordered.empty:
        return ordered
    spread_threshold = gate.get("spread_threshold")
    if spread_threshold is not None:
        first = ordered.groupby("datetime", sort=False).head(1).set_index("datetime")["score"]
        second = ordered.loc[ordered.groupby("datetime").cumcount() == 1].set_index("datetime")["score"]
        spreads = (first - second).dropna()
        ordered = ordered.loc[ordered.datetime.isin(spreads.index[spreads >= float(spread_threshold)])]
    ordered = ordered.loc[ordered.score >= float(gate["threshold"])]
    return ordered.groupby("datetime", sort=False).head(top_n).reset_index(drop=True)
