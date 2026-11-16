"""内部模型族选择的确定性测试，不运行真实外层，也不把模型族测试成绩作为输入。"""

import unittest

import pandas as pd

from scripts.run_family_selected_path_validation import select_family_rule


class FamilyPathSelectionTests(unittest.TestCase):
    """验证同一规则可以依内部证据选择不同族，但不放宽正收益片段和全部有波动条件。"""

    def create_inputs(self):
        """返回两个族的冻结赢家与内部验证表，门槛来自父轮，不在族间选择时重算。"""
        rows = [
            {"method":"method1","rule_id":"a","threshold":.002,"mean_sharpe":4.,
             "closed_lots":100,"positive_segments":1,"active_segments":3,"worst_drawdown":-.1},
            {"method":"method2","rule_id":"b","threshold":.003,"mean_sharpe":1.,
             "closed_lots":40,"positive_segments":2,"active_segments":3,"worst_drawdown":-.05}]
        winners = {r["method"]:{"rule_id":r["rule_id"],"threshold":r["threshold"],"profile":r["rule_id"],
            "selection":"父轮内部验证","horizon":3,"initial":20.,"trailing":40.,"max_days":0} for r in rows}
        return pd.DataFrame(rows),winners,{"min_closed_lots":20,"simple_min_positive_segments":2}

    def test_high_mean_cannot_replace_positive_segments(self):
        """均值更高但只一段盈利的族不能越过原协议，选中族保留原门槛。"""
        table,winners,cfg = self.create_inputs()
        method,rule,metrics = select_family_rule(table,winners,cfg,3)
        self.assertEqual(method,"method2")
        self.assertEqual(rule["threshold"],.003)
        self.assertTrue(metrics["enabled"])

    def test_choice_ignores_outer_columns_and_row_order(self):
        """额外伪造外层收益列及行序不能改变内部选择；没有按年份切换的参数。"""
        table,winners,cfg = self.create_inputs()
        expected = select_family_rule(table,winners,cfg,3)
        poisoned = table.assign(test_sharpe=[1000,-1000],test_return=[999,-999]).iloc[::-1]
        self.assertEqual(expected,select_family_rule(poisoned,winners,cfg,3))
        table.loc[0,"positive_segments"] = 2
        self.assertEqual(select_family_rule(table,winners,cfg,3)[0],"method1")

    def test_cash_and_missing_segment_are_not_dropped(self):
        """任一片段无波动即无效，不能通过只平均活跃片段使某族合格。"""
        table,winners,cfg = self.create_inputs()
        table.loc[1,"active_segments"] = 2
        _,rule,metrics = select_family_rule(table,winners,cfg,3)
        self.assertFalse(metrics["enabled"])
        self.assertEqual(rule["threshold"],1e9)
        self.assertIsNone(metrics["mean_sharpe"])

    def test_parent_gate_mismatch_rejected(self):
        """拒绝更换父轮已冻结候选或门槛，不能在新增选择层暗中调整开仓阈值。"""
        table,winners,cfg = self.create_inputs()
        winners["method2"]["threshold"] = .004
        with self.assertRaises(ValueError):
            select_family_rule(table,winners,cfg,3)


if __name__=="__main__":
    unittest.main()
