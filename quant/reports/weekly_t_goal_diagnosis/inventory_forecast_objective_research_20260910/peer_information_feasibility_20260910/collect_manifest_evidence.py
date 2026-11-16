"""只读v14/v17核心模型manifest与源码，归档候选覆盖率，不读取行情或评价收益。"""
from pathlib import Path
import collections
import csv
import hashlib
import json

import pandas as pd


def main():
    """按显式Huber→L2模型映射核对48份manifest，保留原CSV精确覆盖率文本。"""
    root = Path(__file__).resolve().parents[4]
    robust = root / "reports/inventory_robust_t/run_20260910_172815_737873"
    learning = root / "reports/inventory_learning_t/run_20260910_151232_841321"
    out = Path(__file__).resolve().parent
    inputs, models, records, raw_coverages = {}, [], [], []

    def bind(path):
        """记录本次读取文件的SHA，不改变输入。"""
        path = Path(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        inputs[str(path)] = digest
        return digest

    keys = ["f_return_15m", "f_timing_market_return_15m", "f_timing_excess_return_15m"]
    frequency = collections.Counter()
    for path in sorted(robust.glob("year_*/models/*/feature_manifest.csv")):
        source = learning / path.relative_to(robust).parts[0] / "models" / path.parent.name.replace("huber_", "l2_")
        bind(path)
        bind(source / "feature_manifest.csv")
        current = pd.read_csv(path, float_precision="round_trip")
        original = pd.read_csv(source / "feature_manifest.csv", float_precision="round_trip")
        pd.testing.assert_frame_equal(current, original, check_exact=True)
        metadata = []
        for parent in (path.parent, source):
            bind(parent / "model_metadata.json")
            metadata.append(json.loads((parent / "model_metadata.json").read_text()))
        model, old = metadata
        assert model["features"] == old["features"] == current.loc[current.selected, "feature"].tolist()
        assert model["train_years"] == old["train_years"] and model["excluded_years"] == old["excluded_years"]
        assert model["training_rows"] == old["training_rows"]
        model_id = "/".join(path.relative_to(robust).parts[:-1])
        frequency.update(model["features"])
        models.append(dict(model=model_id, train_years=model["train_years"], excluded_years=model["excluded_years"],
            selected_count=len(model["features"]), candidate_count=len(current), features=model["features"],
            training_rows=model["training_rows"], raw_target_unchanged=True,
            manifest_sha256=inputs[str(path)], source_manifest_sha256=inputs[str(source / "feature_manifest.csv")]))
        for row in current.to_dict("records"):
            records.append(dict(model=model_id, **row))
        with path.open(newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                if row["feature"] in keys:
                    raw_coverages.append(dict(model=model_id, training_rows=model["training_rows"], **row))
    assert len(models) == 24 and len(raw_coverages) == 72
    assert all(row["selected_count"] == 30 and row["candidate_count"] == 38 for row in models)
    cfg_path = robust / "resolved_config.json"
    bind(cfg_path)
    cfg = json.loads(cfg_path.read_text())
    cache = Path(cfg["market_cache_dir"])
    bind(cache / "metadata.json")
    cache_metadata = json.loads((cache / "metadata.json").read_text())
    sources = ["quant_backtest/weekly_t_data.py", "quant_backtest/weekly_t_features.py",
        "quant_backtest/weekly_t_timing.py", "quant_backtest/weekly_t_provenance.py",
        "quant_backtest/weekly_t_short_horizon.py", "quant_backtest/weekly_t_models.py",
        "quant_backtest/weekly_t_timing_data.py", "quant_backtest/inventory_t_learning_data.py",
        "quant_backtest/inventory_t_frozen_cohort.py", "quant_backtest/inventory_t_learning_models.py",
        "quant_backtest/inventory_t_robust_models.py"]
    for name in sources:
        bind(root / name)
    bind(robust / "source_hashes.json")
    frozen = json.loads((robust / "source_hashes.json").read_text())
    assert len(frozen) == 56
    for name, digest in frozen.items():
        assert bind(root / name) == bind(robust / "source_snapshot" / name) == digest
    result = dict(passed=True, scope="core_manifests_and_source_only", no_market_price_or_evaluation_read=True,
        no_fit_or_feature_construction=True, no_rule_change=True, model_count=24, source_model_count=24,
        model_summaries=models, candidate_count=38, selected_union_count=len(frequency),
        selected_frequency=dict(sorted(frequency.items())), coverage_records=raw_coverages,
        candidate_manifest_records=records, feature_min_coverage=cfg["feature_min_coverage"],
        feature_max_correlation=cfg["feature_max_correlation"], t_max_features=cfg["t_max_features"],
        signal_times=cfg["signal_times"], cache_metadata={key: cache_metadata[key] for key in [
            "symbol_count", "weekly_rows", "intraday_rows", "daily_rows", "calendar", "price_adjustment", "universe_rule"]},
        cache_path=str(cache), cache_file_sizes={name: (cache / name).stat().st_size for name in ["daily.pkl", "intraday.pkl", "weekly.pkl"]},
        runtime_sources_unchanged=56, papers=[
            {"url": "https://arxiv.org/html/2201.08283v1", "title": "Lead-lag detection and network clustering for multivariate time series with an application to the US equity market", "sections_read": "abstract; 6.1; 6.2 cost limitation"},
            {"url": "https://arxiv.org/html/1401.0462v1", "title": "Emergence of statistically validated financial intraday lead-lag relationships", "sections_read": "abstract; multiple testing; synchronous versus lagged networks"}],
        implementation_documentation={"url": "https://lightgbm.readthedocs.io/en/stable/Advanced-Topics.html", "section": "Missing Value Handle"},
        inputs_sha256=inputs, evidence_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    destination = out / "core_manifest_evidence.json"
    assert not destination.exists()
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    pd.DataFrame(raw_coverages).to_csv(out / "all_24_models_15m_coverage.csv", index=False)
    pd.DataFrame(records).to_csv(out / "all_24_core_candidate_features.csv", index=False)
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest for path, digest in inputs.items())
    print(json.dumps({"passed": True, "models": len(models), "coverage_records": len(raw_coverages),
        "read_inputs": len(inputs), "evidence_sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
