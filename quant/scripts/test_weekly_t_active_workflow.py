"""验证v9复用来源、真实候选选择与先冻结后外评的工作流，不训练或运行真实账户。

从quant环境执行。创建微型临时v8模型/审计夹具，原生产数据与报告只用于读取
默认配置；对文件身份、错年、错误配置和运行顺序进行有界检查。
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_weekly_t_active as runner
from scripts.test_weekly_t_activation import make_candidate


def sha(path: Path) -> str:
    """计算测试夹具身份，不依赖生产审计实现。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, value) -> None:
    """只写测试拥有的临时JSON。"""
    path.write_text(json.dumps(value), encoding="utf-8")


class TestActiveWorkflow(unittest.TestCase):
    """用临时模型替身验证复用合同，账户与大型行情加载均不执行。"""

    def setUp(self):
        """建立四折微型v8源，真实选择器应选负收益但已完整交易的候选。"""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.cfg = runner.load_configuration(["--test-years", "2023,2026"])
        self.cfg.update(source_training_report_dir=str(self.source), output_dir=str(self.root / "runs"))
        self.previous = deepcopy(self.cfg)
        self.previous.update(research_version="weekly_t_v8_residual_zero_market", test_years=[2023,2024,2025,2026],
                             candidate_support_requirements={"min_complete_pairs":100})
        for key in ("candidate_activation_policy", "source_training_report_dir", "dense_recovery_cache_paths"):
            self.previous.pop(key, None)
        dump(self.source / "resolved_config.json", self.previous)
        dependency = self.root / "frozen_dependency.py"
        dependency.write_text('"""微型依赖快照。"""\n')
        snapshot = self.source / "source_snapshot"
        snapshot.mkdir()
        (snapshot / dependency.name).write_bytes(dependency.read_bytes())
        dump(self.source / "source_hashes.json", {dependency.name:sha(dependency)})
        dump(self.source / "all_rules_frozen_before_outer_test.json", {"test_years":[2023,2024,2025,2026]})
        self.audit = {"passed":True,"complete":True,"full_configured_scope_complete":True,
                      "strict_feature_and_parameter_year_isolation_passed":True,
                      "configured_years":[2023,2024,2025,2026],"requested_years":[2023,2024,2025,2026],
                      "models":{},"supports":{}}
        for year in (2023,2024,2025,2026):
            fold_dir = self.source / f"year_{year}"
            fold_dir.mkdir()
            train = [y for y in (2023,2024,2025,2026) if y != year]
            inner = max(train)
            core = [y for y in train if y != inner]
            fold = {"test_year":year,"train_years":train,"inner_year":inner,"core_years":core,
                    "forward_only":max(train)<year,"outer_t_refit":False}
            dump(fold_dir / "split.json", fold)
            rows = [make_candidate(0,0.,0), make_candidate(1,-3.,2),
                    make_candidate(2,1000.,200,recovery_failure_days=1)]
            for row in rows:
                row.update(test_year=year,inner_year=inner,direction="both",fraction=.25,
                           decision_sha256=str(row["candidate_index"])*64)
            pd.DataFrame(rows).to_csv(fold_dir / "validation_rule_metrics.csv",index=False)
            model = fold_dir / "inner_t_h20_l2_depth4"
            model.mkdir()
            metadata = {"train_years":core,"excluded_years":sorted([year,inner]),"profile":"l2_depth4",
                        "resolved_config":{**deepcopy(self.previous),"t_label_horizon_minutes":20,"t_model_profile":"l2_depth4"},
                        "feature_names":["f_x"],"task":"intraday"}
            dump(model / "model_metadata.json", metadata)
            joblib.dump({"metadata":metadata,"features":["f_x"],"models":{"return":"unused_test_stub"}},
                        model / "model_bundle.joblib")
            (model / "return_model.txt").write_text(f"test-native-{year}\n")
            support = fold_dir / "support_h20.json"
            dump(support,{"bounds":{"f_x":{"lower":-10.,"upper":10.}},"training_rows":20})
            self.audit["models"][f"year_{year}/inner_t_h20_l2_depth4"] = {
                "passed":True,"metadata_sha256":sha(model / "model_metadata.json"),
                "model_bundle_sha256":sha(model / "model_bundle.joblib"),"native_model_sha256":sha(model / "return_model.txt")}
            self.audit["supports"][f"year_{year}/support_h20.json"] = {"passed":True,"support_file_sha256":sha(support)}
        dump(self.source / "independent_residual_audit.json",self.audit)
        self.counter = 0

    def output_dir(self) -> Path:
        """每次被测调用使用独立输出，保留所有冻结输入。"""
        self.counter += 1
        out = self.root / f"out_{self.counter}"
        out.mkdir()
        return out

    def test_smoke_only_slices_execution_and_reuse_rejects_economic_change(self):
        """smoke不改费用/周选股/信号配置；已有内层结果不能用于不同成本。"""
        full = runner.load_configuration([])
        smoke = runner.load_configuration(["--max-symbols","8","--smoke-weeks","3","--test-years","2026"])
        changed = {key for key in set(full)|set(smoke) if full.get(key)!=smoke.get(key)}
        self.assertEqual(changed,{"max_symbols","smoke_weeks","test_years"})
        with patch.object(runner,"PROJECT_ROOT",self.root):
            self.assertTrue(runner.validate_training_source(self.cfg)["configuration_compatible"])
            for key,value in (("commission_rate",.001),("weekly_return_weight",.8),("min_price",4.)):
                cfg = {**self.cfg,key:value}
                with self.subTest(key=key), self.assertRaisesRegex(ValueError,"配置不一致"):
                    runner.validate_training_source(cfg)

    def test_source_audit_must_be_successful_and_complete(self):
        """存在失败或未完成的审计文件，不能被当作通过审计的训练来源。"""
        for key in ("passed","complete","full_configured_scope_complete"):
            audit = deepcopy(self.audit)
            audit[key] = False
            dump(self.source / "independent_residual_audit.json",audit)
            with self.subTest(key=key), patch.object(runner,"PROJECT_ROOT",self.root), self.assertRaises(ValueError):
                runner.validate_training_source(self.cfg)
        dump(self.source / "independent_residual_audit.json",self.audit)

    def test_freeze_preserves_exact_core_weights_and_selects_real_negative_candidate(self):
        """没有原统计合格者仍选真实有交易的亏损候选，所有折一并冻结。"""
        out = self.output_dir()
        folds, table = runner.freeze_active_rules(self.cfg,out,{})
        self.assertEqual([f["test_year"] for f in folds],[2023,2026])
        self.assertEqual(table.loc[table.selected,"candidate_index"].tolist(),[1,1])
        self.assertFalse(table.strict_eligible.any())
        marker = json.loads((out/"all_rules_frozen_before_outer_test.json").read_text())
        self.assertEqual(marker["test_years"],[2023,2026])
        self.assertTrue(marker["all_t_enabled"])
        for year in (2023,2026):
            rule = json.loads((out/f"year_{year}/selected_rule.json").read_text())
            self.assertIs(rule["enabled"],True)
            self.assertEqual(rule["inner_incremental_net_pnl"],-3.)
            self.assertEqual(rule["inner_complete_pairs"],2)
            source = self.source/f"year_{year}/inner_t_h20_l2_depth4/model_bundle.joblib"
            self.assertEqual(sha(source),sha(out/f"year_{year}/t_model/model_bundle.joblib"))
            self.assertEqual(rule["model_bundle_sha256"],sha(source))

    def test_candidate_fold_and_model_year_mismatch_rejected(self):
        """另一折候选或错误核心模型不能因为收益高而被复用。"""
        metrics = self.source/"year_2023/validation_rule_metrics.csv"
        original_metrics = metrics.read_bytes()
        rows = pd.read_csv(metrics)
        rows["inner_year"] = 2025
        rows.to_csv(metrics,index=False)
        with self.assertRaisesRegex(ValueError,"候选评价年份"):
            runner.freeze_active_rules(self.cfg,self.output_dir(),{})
        metrics.write_bytes(original_metrics)
        model = self.source/"year_2023/inner_t_h20_l2_depth4/model_metadata.json"
        metadata = json.loads(model.read_text())
        metadata["train_years"] = [2023,2025]
        dump(model,metadata)
        with self.assertRaises(ValueError):
            runner.freeze_active_rules(self.cfg,self.output_dir(),{})

    def test_model_and_support_files_must_match_completed_audit(self):
        """元数据/权重/原生模型/支持域被替换，不能重新记录新SHA后继续复用。"""
        folder = self.source/"year_2023"
        files = [folder/"inner_t_h20_l2_depth4"/name for name in
                 ("model_metadata.json","model_bundle.joblib","return_model.txt")]
        files.append(folder/"support_h20.json")
        for path in files:
            before = path.read_bytes()
            if path.name == "model_metadata.json":
                metadata = json.loads(before)
                metadata["resolved_config"]["t_model"]["n_estimators"] += 1
                dump(path,metadata)
            else:
                path.write_bytes(before + b" ")
            with self.subTest(file=path.name), self.assertRaises(ValueError):
                runner.freeze_active_rules(self.cfg,self.output_dir(),{})
            path.write_bytes(before)

    def test_outer_refuses_incomplete_freeze_or_changed_rule_before_any_account(self):
        """一个未冻结或被改写的后续折，必须在首折外层账户开始前阻止运行。"""
        out = self.output_dir()
        with patch.object(runner,"run_weekly_portfolio") as account:
            with self.assertRaises((FileNotFoundError,ValueError)):
                runner.evaluate_active_folds([],pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),self.cfg,out,{})
            account.assert_not_called()
        folds,_ = runner.freeze_active_rules(self.cfg,out,{})
        marker_path = out/"all_rules_frozen_before_outer_test.json"
        marker = json.loads(marker_path.read_text())
        bad = deepcopy(marker)
        bad["test_years"] = [2023]
        dump(marker_path,bad)
        with patch.object(runner,"run_weekly_portfolio") as account:
            with self.assertRaises(ValueError):
                runner.evaluate_active_folds(folds,pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),self.cfg,out,{})
            account.assert_not_called()
        dump(marker_path,marker)
        rule = out/"year_2026/selected_rule.json"
        rule.write_bytes(rule.read_bytes()+b" ")
        with patch.object(runner,"run_weekly_portfolio") as account:
            with self.assertRaisesRegex(ValueError,"冻结规则"):
                runner.evaluate_active_folds(folds,pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),self.cfg,out,{})
            account.assert_not_called()

    def test_main_freezes_all_rules_before_loading_or_evaluating_market(self):
        """入口先完成全部内层选择，再加载/推断外层；报告阶段不回选。"""
        events = []
        original_freeze = runner.freeze_active_rules
        def freeze(cfg,directory,inputs):
            result = original_freeze(cfg,directory,inputs)
            events.append("all_frozen")
            return result
        def market(cfg,directory,inputs):
            marker = json.loads((directory/"all_rules_frozen_before_outer_test.json").read_text())
            self.assertEqual(marker["test_years"],cfg["test_years"])
            events.append("market")
            return pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
        def evaluate(folds,universe,intraday,daily,cfg,directory,inputs):
            self.assertEqual(events,["all_frozen","market"])
            events.append("outer")
            return pd.DataFrame({"test_year":cfg["test_years"]}),[]
        def snapshot(directory):
            dump(directory/"source_hashes.json",{})
        with patch.object(runner,"load_configuration",return_value=self.cfg), \
             patch.object(runner,"validate_training_source",return_value={"passed":True}), \
             patch.object(runner,"snapshot_active_sources",side_effect=snapshot), \
             patch.object(runner,"freeze_active_rules",side_effect=freeze), \
             patch.object(runner,"load_frozen_market_data",side_effect=market), \
             patch.object(runner,"evaluate_active_folds",side_effect=evaluate), \
             patch.object(runner,"write_active_report",side_effect=lambda *args:events.append("report")):
            runner.main([])
        self.assertEqual(events,["all_frozen","market","outer","report"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
