"""CTA开仓时点候选：只改变当天已完成采样点的检查时钟，不移动特征或成交时间。

输入为原验证预测和固定规则，输出对齐开仓信号；每批仓位、T+1和退出保持原合同。
"""

import itertools
import numpy as np
import pandas as pd

from .cta_research_data import calculate_identity
from .cta_validation_search import ValidationCtaStrategy


def calculate_entry_clock(times: pd.Series, frequency: int, offset: int) -> np.ndarray:
    """按times上海交易时段计算frequency分钟检查周期和offset起点，仅支持已有24分钟采样网格。"""
    if frequency <= 0 or frequency % 24 or 240 % frequency or offset < 0 or offset >= frequency or offset % 24:
        raise ValueError('频率/偏移不在已缓存24分钟交易网格内')
    minute = times.dt.hour.to_numpy()*60 + times.dt.minute.to_numpy()
    session = np.where(minute <= 690, minute-571, minute-781+120)
    return (session >= 0) & (session < 236) & (session % frequency == offset)


class TimingValidationStrategy(ValidationCtaStrategy):
    """非ST周池的时点CTA开仓，输出只做多信号，不因时钟变化平仓或允许T+0。"""
    name = 'timing_validation_cta'

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """data保持原日期/索引；先应用原模型/趋势条件，再用固定offset筛时点，不伪造早到信号。"""
        base_rule = {**self.rule, 'frequency': 24}
        output = ValidationCtaStrategy(base_rule).generate_signals(data)
        clock = calculate_entry_clock(data.datetime, self.rule['frequency'], self.rule['offset'])
        output['open_signal'] = (output.open_signal.eq(1) & clock).astype(int)
        return output


def create_timing_rules(cfg: dict) -> list[dict]:
    """根据cfg时点候选生成全年共用规则，模型/门槛/时钟不允许包含年份条件。"""
    rules = []
    for name, style, offset, gate in itertools.product(cfg['timing_model_names'], cfg['styles'], cfg['offsets'], cfg['probability_gates']):
        rule = dict(model_name=name, exit_id=int(name.split('_')[0][4:]), style=style, context='none',
                    frequency=240, offset=offset, gate=gate)
        rule['rule_id'] = calculate_identity(rule)
        rules.append(rule)
    return rules
