"""验证主动T的内部选择规则；纯候选表测试，不读取真实收益或运行账户。

在quant环境从项目目录执行。覆盖亏损低样本、零交易、五类执行失败、
数值完整性、并列稳定性和输入不变；不触及原统计门控或冻结实验源。
"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quant_backtest.weekly_t_activation import EXECUTION_FAILURE_COLUMNS, select_active_candidate


def make_candidate(index: int, net: float, pairs: int = 1, **changes) -> dict:
    """构造执行合格但原统计门槛未过的内部候选，支持定点覆盖边界。"""
    row = {"candidate_index": index, "incremental_net_pnl": net, "t_pair_count": pairs,
           "completed_pair_count": pairs, **{column: 0 for column in EXECUTION_FAILURE_COLUMNS},
           "strict_eligible": False, "rejection_reasons": "nonpositive_weekly_bootstrap_lower|insufficient_complete_pairs",
           "active_days": 1, "active_weeks": 1, "active_symbols": 1,
           "weekly_increment_ci_low_rmb": -1000., "selected": False,
           "horizon_minutes": 20, "profile": "l2_depth4", "net_margin": .002, "enabled": True}
    row.update(changes)
    return row


class TestActiveCandidateSelection(unittest.TestCase):
    """检查真实交易优先的选择合同，不以样本置信度或盈利正负关闭T。"""

    def test_loss_low_sample_candidate_beats_zero_trade_placeholder(self):
        """零交易的零收益不能击败真实交易的内层亏损候选。"""
        frame = pd.DataFrame([make_candidate(0, 0., 0), make_candidate(1, -21.93, 2), make_candidate(2, -100., 5)])
        chosen, annotated = select_active_candidate(frame)
        self.assertEqual(chosen["candidate_index"], 1)
        self.assertEqual(chosen["incremental_net_pnl"], -21.93)
        self.assertFalse(chosen["strict_eligible"])
        self.assertEqual(annotated.active_execution_eligible.tolist(), [False, True, True])
        self.assertEqual(annotated.execution_rejection_reasons.iloc[0], "no_completed_t_pairs")

    def test_each_execution_failure_blocks_higher_profit(self):
        """任何一项用户执行要求失败都不能靠更高内层利润晋级。"""
        for column in EXECUTION_FAILURE_COLUMNS:
            with self.subTest(column=column):
                frame = pd.DataFrame([make_candidate(0, 10000., 300, **{column: 1}), make_candidate(1, -5.)])
                chosen, annotated = select_active_candidate(frame)
                self.assertEqual(chosen["candidate_index"], 1)
                self.assertEqual(annotated.execution_rejection_reasons.iloc[0], column)

    def test_statistics_and_old_selected_never_control_choice(self):
        """CI、样本量、集中度和旧selected均仅供报告，不作选择输入。"""
        frame = pd.DataFrame([make_candidate(4, 220.21, 15, weekly_increment_ci_low_rmb=-99999.,
                                                max_day_pair_fraction=1., active_days=1),
                              make_candidate(1, 100., 1000, strict_eligible=True, selected=True,
                                             weekly_increment_ci_low_rmb=10.)])
        chosen, annotated = select_active_candidate(frame)
        self.assertEqual(chosen["candidate_index"], 4)
        self.assertEqual(annotated.selected.tolist(), [True, False])
        self.assertEqual(annotated.strict_eligible.tolist(), [False, True])

    def test_tie_uses_candidate_index_not_row_order(self):
        """相同利润选较小编号，输入倒序和重复DataFrame索引不影响结果。"""
        frame = pd.DataFrame([make_candidate(8, 36.5), make_candidate(2, 36.5), make_candidate(3, 36.4)], index=[7, 7, 1])
        for candidate in (frame, frame.iloc[::-1]):
            chosen, annotated = select_active_candidate(candidate)
            self.assertEqual(chosen["candidate_index"], 2)
            self.assertEqual(annotated.index.tolist(), candidate.index.tolist())
            self.assertEqual(annotated.selected.sum(), 1)

    def test_source_input_and_original_statistics_unchanged(self):
        """添加审计列不改变原索引、行序、统计结论或输入对象。"""
        frame = pd.DataFrame([make_candidate(3, 5.), make_candidate(1, -4., 0)], index=[99, 10])
        before = frame.copy(deep=True)
        chosen, annotated = select_active_candidate(frame)
        pd.testing.assert_frame_equal(frame, before)
        pd.testing.assert_frame_equal(annotated.drop(columns=["active_execution_eligible", "execution_rejection_reasons", "selected"]),
                                      before.drop(columns="selected"))
        self.assertTrue(chosen["selected"])
        annotated.loc[99, "incremental_net_pnl"] = 999.
        pd.testing.assert_frame_equal(frame, before)

    def test_missing_fields_and_duplicate_columns_raise(self):
        """缺少审计字段不能被默认0掩盖；重复列同样拒绝。"""
        frame = pd.DataFrame([make_candidate(0, 1.)])
        for column in ("candidate_index", "incremental_net_pnl", "t_pair_count", *EXECUTION_FAILURE_COLUMNS):
            with self.subTest(column=column), self.assertRaisesRegex(ValueError, "缺少必要字段"):
                select_active_candidate(frame.drop(columns=column))
        with self.assertRaisesRegex(ValueError, "重复列名"):
            select_active_candidate(pd.concat([frame, frame[["t_pair_count"]]], axis=1))

    def test_required_numeric_fields_reject_invalid_values(self):
        """非有限、布尔、字符串及非法计数不被悄悄当作合法审计值。"""
        for column in ("candidate_index", "incremental_net_pnl", "t_pair_count", *EXECUTION_FAILURE_COLUMNS):
            for value in (np.nan, np.inf, -np.inf, True, "0", None):
                with self.subTest(column=column, value=value), self.assertRaises(ValueError):
                    select_active_candidate(pd.DataFrame([make_candidate(0, 1., **{column: value})]))
        for column in ("candidate_index", "t_pair_count", *EXECUTION_FAILURE_COLUMNS):
            for value in (-1, .5):
                with self.subTest(column=column, value=value), self.assertRaisesRegex(ValueError, "非负整数"):
                    select_active_candidate(pd.DataFrame([make_candidate(0, 1., **{column: value})]))

    def test_completed_pair_consistency_and_optional_column(self):
        """旧表可没有别名，但有完整配对别名时必须逐行一致且合法。"""
        chosen, _ = select_active_candidate(pd.DataFrame([make_candidate(0, 1.)]).drop(columns="completed_pair_count"))
        self.assertEqual(chosen["t_pair_count"], 1)
        for value in (0, 2, np.nan, np.inf, -.5, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                select_active_candidate(pd.DataFrame([make_candidate(0, 1., completed_pair_count=value)]))

    def test_no_active_execution_valid_candidate_raises(self):
        """所有候选无实际交易或执行失败时显式报错，不返回关闭或伪交易方案。"""
        frames = [pd.DataFrame([make_candidate(0, 0., 0)]),
                  pd.DataFrame([make_candidate(0, 100., recovery_failure_days=1), make_candidate(1, 0., 0)])]
        for frame in frames:
            with self.subTest(rows=len(frame)), self.assertRaisesRegex(ValueError, "没有完成真实T"):
                select_active_candidate(frame)
        with self.assertRaisesRegex(ValueError, "没有内部候选"):
            select_active_candidate(pd.DataFrame())

    def test_candidate_identity_must_be_unique(self):
        """候选编号重复不能靠DataFrame输入顺序破坏并列规则。"""
        with self.assertRaisesRegex(ValueError, "candidate_index必须唯一"):
            select_active_candidate(pd.DataFrame([make_candidate(2, 1.), make_candidate(2, 2.)]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
