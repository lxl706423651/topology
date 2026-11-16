"""冻结训练审计的合成回归测试；只读内存候选表，不加载真实模型或测试结果。"""

import unittest

import pandas as pd

from scripts.audit_multiscale_frozen_training import validate_threshold_contract


class FrozenTrainingAuditTests(unittest.TestCase):
    """校准、验证、冻结门槛必须按方法对齐，现金规则也必须可追溯。"""

    def setUp(self):
        """构造同名但不同方法门槛，模拟真实CTA叠加的分数差异。"""
        self.training = pd.DataFrame({"method": ["method1", "method2"], "rule_id": ["same", "same"], "threshold": [.01, .02]})
        self.validation = self.training.copy()
        self.rules = {"method1": {"rule_id": "same", "threshold": .01}, "method2": {"rule_id": "same", "threshold": .02}}

    def test_distinct_thresholds_pass(self):
        """不同方法不要求相同门槛，只要求各自跨阶段一致。"""
        self.assertEqual(validate_threshold_contract(self.training, self.validation, self.rules), 2)

    def test_overwritten_validation_is_rejected(self):
        """复现方法二覆盖方法一，不允许审计只核对最终胜者而漏掉候选。"""
        self.validation.loc[0, "threshold"] = .02
        with self.assertRaisesRegex(ValueError, "校准与验证"):
            validate_threshold_contract(self.training, self.validation, self.rules)

    def test_changed_frozen_value_is_rejected(self):
        """即使校准和验证一致，另改冻结门槛也要报错。"""
        self.rules["method1"]["threshold"] = .03
        with self.assertRaisesRegex(ValueError, "冻结门槛"):
            validate_threshold_contract(self.training, self.validation, self.rules)

    def test_missing_or_duplicate_candidate_is_rejected(self):
        """每个验证候选必须恰好对应一条方法内校准记录。"""
        with self.assertRaisesRegex(ValueError, "没有对应"):
            validate_threshold_contract(self.training.iloc[:1], self.validation, self.rules)
        with self.assertRaisesRegex(ValueError, "不唯一"):
            validate_threshold_contract(pd.concat([self.training, self.training]), self.validation, self.rules)

    def test_cash_still_audits_candidates(self):
        """现金结论不能掩盖候选契约错误，也不能随意改现金哨兵。"""
        rules = {m: {"rule_id": "cash_validation_failed", "threshold": 1e9} for m in self.rules}
        self.assertEqual(validate_threshold_contract(self.training, self.validation, rules), 2)
        rules["method1"]["threshold"] = 100
        with self.assertRaisesRegex(ValueError, "现金保护"):
            validate_threshold_contract(self.training, self.validation, rules)


if __name__ == "__main__":
    unittest.main()
