"""用人工校准表检查复现审计和现金年晋级规则，不访问真实外层预测或收益。"""

import unittest

import pandas as pd

from scripts.audit_exit_alignment_comparison import (
    CONTROL_FIELDS, CONTROL_NAMES, calculate_promotion_status, compare_calibration_controls,
)


def create_control_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """构造两方法、两门槛退出的人工新旧校准对照，字段均有限且逐对照相同。"""
    old, new = [], []
    for method in ["method1", "method2"]:
        for name, previous in CONTROL_NAMES.items():
            row = dict.fromkeys(CONTROL_FIELDS, 1.0)
            row.update(method=method, profile="test_profile", quantile=.8)
            old.append({**row, "name": previous, "rule_id": f"{previous}_q0.8"})
            new.append({**row, "name": name, "rule_id": f"{name}_q0.8"})
    return pd.DataFrame(old), pd.DataFrame(new)


class AlignmentAuditTests(unittest.TestCase):
    """不能用缺行、浮点容差或删除现金年制造复现和晋级。"""

    def test_identical_controls_pass(self):
        """规则代码允许更名，真正比较键和数值必须一致；忽略非对照退出。"""
        old, new = create_control_tables()
        extra = new.iloc[[0]].assign(name="matched_horizon", rule_id="not_a_control")
        result = compare_calibration_controls(old, pd.concat([new, extra], ignore_index=True))
        self.assertEqual(len(result), 4)
        self.assertTrue((result.filter(like="差值") == 0).all().all())

    def test_metric_or_gate_change_rejected(self):
        """门槛或任一数值变更不能被相近值掩盖。"""
        for field in CONTROL_FIELDS:
            old, new = create_control_tables()
            new.loc[0, field] += 1e-12
            with self.assertRaisesRegex(ValueError, "未精确复现"):
                compare_calibration_controls(old, new)

    def test_missing_duplicate_or_nan_rejected(self):
        """缺少对照、重复键、NaN均是审计失败，不应自动丢行。"""
        old, new = create_control_tables()
        for invalid in [new.iloc[1:], pd.concat([new, new.iloc[[0]]], ignore_index=True),
                        new.assign(threshold=float("nan"))]:
            with self.assertRaises(ValueError):
                compare_calibration_controls(old, invalid)

    def test_cash_year_blocks_method_promotion(self):
        """某方法一折现金即不完整，另一方法可独立满足必要条件，但不是盈利结论。"""
        years, methods = [2023, 2024, 2025, 2026], ["method1", "method2"]
        folds = {str(y): {"rules": {m: {"rule_id": "enabled"} for m in methods}} for y in years}
        folds["2025"]["rules"]["method1"]["rule_id"] = "cash_validation_failed"
        result = calculate_promotion_status(folds, years, methods).set_index("方法代码")
        self.assertFalse(result.loc["method1", "全部折内部启用"])
        self.assertEqual(result.loc["method1", "现金保护年份"], "2025")
        self.assertTrue(result.loc["method2", "全部折内部启用"])
        del folds["2025"]
        with self.assertRaises(ValueError):
            calculate_promotion_status(folds, years, methods)

    def test_incomplete_methods_rejected(self):
        """不能只核对可盈利或非现金的方法。"""
        with self.assertRaises(ValueError):
            calculate_promotion_status({"2023": {"rules": {"method1": {"rule_id": "enabled"}}}},
                                       [2023], ["method1", "method2"])


if __name__ == "__main__":
    unittest.main()
