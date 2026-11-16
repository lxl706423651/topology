"""独立核验固定底仓T报告的已保存账簿、原始分钟和冻结身份，不调用账户引擎。

必须在主进程结束且runtime_integrity_audit.json落盘后运行。输入--report-dir；
新建独立审计目录，默认不覆盖任何原结果。实际当前源码允许已声明的后续开发，
原运行身份以报告source_snapshot为准；失败也保存脚本、输入SHA和错误信息。
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quant_backtest.weekly_t_data import load_weekly_symbol_bars

FAILURE_KEYS = ['inventory_mismatch_days','recovery_failure_days','pending_pairs',
                't1_violation_count','base_plan_mismatch_count']


def digest(path):
    """流式计算文件SHA，避免大型输入占用额外内存。"""
    h = hashlib.sha256()
    with open(path,'rb') as f:
        for part in iter(lambda:f.read(8<<20),b''):
            h.update(part)
    return h.hexdigest()


def require(condition, message):
    """断言真实业务不变量，任何失败都不能被统计平均掩盖。"""
    if not bool(condition):
        raise AssertionError(message)


def close(actual, expected, label, atol=1e-6):
    """金额容差固定一百万分之一元；不放宽成交资格或价格阈值。"""
    require(np.isfinite(actual) and np.isfinite(expected) and abs(actual-expected)<=atol,
            f'{label}: actual={actual!r}, expected={expected!r}')


def parse_time(value):
    """兼容CSV中整日期与完整时分秒混排，保持所有原始时间含义。"""
    return pd.to_datetime(value,format='mixed',errors='raise')


def truth(value):
    """严格解析CSV布尔列，不将非空False字符串当True。"""
    if isinstance(value,(bool,np.bool_)):
        return bool(value)
    if value in ('True','true'):
        return True
    if value in ('False','false'):
        return False
    raise AssertionError(f'非法布尔:{value!r}')


def next_minute(timestamp):
    """独立以交易分钟时钟计算延迟，午间不累计休市分钟。"""
    stamp=pd.Timestamp(timestamp); minute=stamp.hour*60+stamp.minute
    if minute==690:
        minute=781
    elif 570<=minute<690 or 780<=minute<900:
        minute+=1
    else:
        return None
    return stamp.normalize()+pd.Timedelta(minutes=minute)


def fee(notional, side, timestamp, cfg):
    """独立复算手续费，卖出印花税按实际成交日期切换。"""
    commission=max(notional*cfg['commission_rate'],cfg['min_commission'])
    transfer=notional*cfg['transfer_rate']
    stamp=cfg['stamp_rate_before_20230828'] if timestamp<pd.Timestamp('2023-08-28') else cfg['stamp_rate_from_20230828']
    return commission+transfer+(notional*stamp if side=='sell' else 0.)


class Auditor:
    """持有只读文件身份和原CSV缓存；账户账簿完全由本类独立计算。"""

    def __init__(self,report,out):
        self.report,self.out=report,out
        self.inputs={};self.raw={};self.raw_files={};self.results=[]
        self.cfg=self.json(report/'resolved_config.json')
        market_path=Path(self.cfg['market_cache_dir'])/'daily.pkl'
        if not market_path.is_absolute():market_path=ROOT/market_path
        market=pd.read_pickle(self.record(market_path))
        self.market_calendar=pd.DatetimeIndex(parse_time(market.date).unique()).normalize().sort_values()
        self.checked_raw_events=0

    def record(self,path):
        """登记每个实际读取文件的初始字节，审计结束后再次核对。"""
        path=Path(path).resolve()
        if str(path) not in self.inputs:
            self.inputs[str(path)]=digest(path)
        return path

    def json(self,path):
        """读取有身份的JSON文件。"""
        return json.loads(self.record(path).read_text())

    def csv(self,path):
        """严格保留浮点CSV往返数值；真正空表仅限零字节表体。"""
        path=self.record(path)
        try:
            return pd.read_csv(path,float_precision='round_trip')
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    def bars(self,year,symbol):
        """仅加载当前报告成交/持仓涉及股票年的原CSV，复用已冻结合法行情读取器。"""
        key=(int(year),str(symbol))
        if key not in self.raw:
            path=Path(self.cfg['data_root'])/str(year)/f'{symbol}.csv'
            if not path.is_absolute():path=ROOT/path
            self.record(path)
            frame,audit=load_weekly_symbol_bars([str(path)],f'{year}-01-01',f'{year}-12-31')
            self.raw[key]=frame.set_index('datetime')
            self.raw_files[str(path.resolve())]={'sha256':self.inputs[str(path.resolve())],**audit}
        return self.raw[key]

    def raw_quote(self,timestamp,symbol):
        """09:30开盘对应09:31bar的open，其他成交对应当前分钟close。"""
        stamp=pd.Timestamp(timestamp)
        opening=stamp.hour==9 and stamp.minute==30
        raw_time=stamp+pd.Timedelta(minutes=1) if opening else stamp
        raw=self.bars(stamp.year,symbol)
        require(raw_time in raw.index,f'原始成交分钟缺失:{symbol}/{raw_time}')
        bar=raw.loc[raw_time]
        require(bar.volume>0 and bar.amount>0,f'无量原始分钟出现成交:{symbol}/{raw_time}')
        return raw_time,float(bar['open' if opening else 'close']),float(bar.amount)

    def verify_event(self,event,used):
        """逐成交核原价、滑点、费用、整手和同账户同原始分钟累计容量。"""
        cfg=self.cfg;qty=float(event.shares);stamp=event.datetime
        require(np.isfinite(qty) and qty>0 and qty%cfg['lot_size']==0,'成交必须正整手')
        require(event.side in ('buy','sell'),'成交方向非法')
        raw_time,reference,amount=self.raw_quote(stamp,event.symbol)
        close(float(event.reference_price),reference,'成交参考价',1e-9)
        price=reference*(1+cfg['slippage_rate'] if event.side=='buy' else 1-cfg['slippage_rate'])
        close(float(event.price),price,'成交滑点价',1e-9)
        close(float(event.notional),qty*price,'成交金额')
        close(float(event.fee),fee(qty*price,event.side,stamp,cfg),'成交费用')
        close(float(event.slippage_cost),qty*abs(price-reference),'滑点成本')
        key=(raw_time,event.symbol);used[key]+=qty*reference
        require(used[key]<=amount*cfg['max_bar_amount_fraction']+1e-6,
                f'同一原分钟累计容量超限:{key}/{used[key]}/{amount}')
        require(stamp>event.signal_datetime,'成交不得早于或等于信号')
        self.checked_raw_events+=1

    def account(self,folder):
        """逐日从两本成交事件重建现金、旧股、配对与原始底仓成本并对账。"""
        cfg=self.cfg
        metrics=self.json(folder/'metrics.json')
        curve=self.csv(folder/'test_equity_curve.csv')
        curve['datetime']=parse_time(curve.datetime)
        days=pd.DatetimeIndex(curve.datetime.dt.normalize())
        require(not days.has_duplicates and days.is_monotonic_increasing,'净值市场日不唯一/未排序')
        require(len(days)>0 and len(set(days.year))==1,'每账户必须属于同一评价年度')
        expected_days=self.market_calendar[self.market_calendar.year==days[0].year]
        if cfg.get('smoke_days',0):expected_days=expected_days[:cfg['smoke_days']]
        require(days.equals(expected_days),'净值未覆盖冻结全市场日历/烟测范围，包括现金期')
        books={}
        for name,file in [('actual','test_trade_events.csv'),('baseline','baseline_trade_events.csv')]:
            events=self.csv(folder/file)
            for col in ('datetime','signal_datetime','planned_recovery_time'):
                if col in events:events[col]=parse_time(events[col])
            require(events.empty or events.datetime.is_monotonic_increasing,'原事件行顺序不是成交时间顺序')
            require(events.empty or events.datetime.dt.normalize().isin(days).all(),'事件超出净值日历')
            books[name]={'events':events,'qty':defaultdict(int),'cash':float(cfg['initial_cash']),
                         'used':defaultdict(float),'groups':{d:p for d,p in events.groupby(events.datetime.dt.normalize())}}
        trips=self.csv(folder/'test_round_trips.csv')
        for col in ('first_datetime','second_datetime','buy_datetime','sell_datetime','planned_recovery_time'):
            if col in trips:trips[col]=parse_time(trips[col])
        audit=self.csv(folder/'daily_inventory_audit.csv')
        audit['date']=parse_time(audit.date)
        require(not audit.duplicated(['date','symbol']).any(),'逐日库存键重复')
        audit_groups={d:p.set_index('symbol') for d,p in audit.groupby('date')}
        positions=self.csv(folder/'test_open_positions.csv')
        self.csv(folder/'position_amount_audit.csv')
        actual_events=books['actual']['events'];base_events=books['baseline']['events']
        base_actual=actual_events.loc[actual_events.event_type.ne('t')]
        require(base_events.empty or base_events.side.eq('buy').all(),'基准出现自动或人工底仓卖出')
        require(base_actual.empty or base_actual.side.eq('buy').all(),'实际出现底仓卖出')
        base_cost=float((base_events.notional+base_events.fee).sum())
        auto=base_events.loc[base_events.base_order_type.eq('automatic')]
        require(auto.symbol.nunique()==len(auto),'自动底仓同股不止首次成功买入')
        require(auto.notional.between(cfg['min_position_notional']-1e-7,cfg['max_position_notional']+1e-7).all(),'自动底仓名义金额超区间')
        auto_actual=base_actual.loc[base_actual.base_order_type.eq('automatic')]
        require(auto_actual.notional.between(cfg['min_position_notional']-1e-7,cfg['max_position_notional']+1e-7).all(),'实际自动底仓金额超区间')
        marks={};first={};recoveries={};active={};complete=set()
        inventory_bad=set();recovery_bad=set();base_mismatch=0
        complete_at={};first_quantities={}
        for ci,day in enumerate(days):
            opening={name:dict(book['qty']) for name,book in books.items()}
            stats={name:{'sold':defaultdict(int),'base_buys':defaultdict(int),'manual':defaultdict(int)} for name in books}
            cycles=defaultdict(int)
            for name,book in books.items():
                old=defaultdict(int,opening[name])
                rows=book['groups'].get(day,pd.DataFrame())
                for event in rows.itertuples(index=False):
                    self.verify_event(event,book['used'])
                    symbol=event.symbol;qty=int(event.shares)
                    close(float(event.sellable_before),old[symbol],'事件前可卖股数',0)
                    if event.side=='buy':
                        book['cash']-=event.notional+event.fee
                        book['qty'][symbol]+=qty
                    else:
                        require(qty<=old[symbol] and qty<=book['qty'][symbol],'T+1超卖或空头')
                        old[symbol]-=qty;book['qty'][symbol]-=qty
                        book['cash']+=event.notional-event.fee
                        stats[name]['sold'][symbol]+=qty
                    require(book['cash']>=-1e-6,'负现金')
                    close(float(event.cash_after),book['cash'],'事件后现金')
                    close(float(event.sellable_after),old[symbol],'事件后可卖股数',0)
                    if event.event_type!='t':
                        stats[name]['base_buys'][symbol]+=qty
                        if event.base_order_type=='manual':stats[name]['manual'][symbol]+=qty
                    elif name=='actual':
                        pair=int(event.pair_id);fragment=int(event.fragment_id)
                        if fragment==0:
                            require(pair not in first and symbol not in active,'首腿pair重复或同股债务重叠')
                            require(event.datetime==next_minute(event.signal_datetime),'首腿不是下一交易分钟')
                            require(opening[name].get(symbol,0)>=qty,'首腿超过日初旧股')
                            require(pd.notna(event.planned_recovery_time) and event.planned_recovery_time>event.datetime,'恢复时钟无效')
                            first[pair]=event;active[symbol]=[pair,qty]
                            cycles[symbol]+=1;first_quantities[pair]=qty
                        else:
                            require(pair in first and active.get(symbol,[None])[0]==pair,'恢复无对应待恢复首腿')
                            original=first[pair]
                            require(event.side!=original.side and event.datetime>=original.planned_recovery_time,'恢复方向或时间错误')
                            require(qty<=active[symbol][1],'恢复超原股数')
                            active[symbol][1]-=qty
                            require((pair,fragment) not in recoveries,'恢复片段重复')
                            recoveries[pair,fragment]=event
                            if active[symbol][1]==0:
                                complete.add(pair);complete_at[pair]=event.datetime;del active[symbol]
            current_audit=audit_groups.get(day,pd.DataFrame())
            symbols=set(opening['actual'])|set(books['actual']['qty'])|set(books['baseline']['qty'])
            for symbol in symbols:
                opened=opening['actual'].get(symbol,0);actual_qty=books['actual']['qty'].get(symbol,0)
                base_qty=books['baseline']['qty'].get(symbol,0);base_open=opening['baseline'].get(symbol,0)
                if not (opened or actual_qty or base_qty):continue
                require(symbol in current_audit.index,'持股日缺库存审计')
                saved=current_audit.loc[symbol];buys=stats['actual']['base_buys'][symbol]
                diff=actual_qty-opened-buys
                fields={'opening_shares':opened,'opening_sellable':opened,'closing_shares':actual_qty,
                        'baseline_opening_shares':base_open,'baseline_closing_shares':base_qty,
                        'weekly_buys':buys,'weekly_sells':0,'target_shares':opened+buys,
                        'inventory_difference':diff,'baseline_inventory_difference':actual_qty-base_qty,
                        'day_sold':stats['actual']['sold'][symbol],
                        'closing_sellable':opened-stats['actual']['sold'][symbol],
                        'cycles_started':cycles[symbol],
                        'manual_buys':stats['actual']['manual'][symbol],
                        'base_buys':buys-stats['actual']['manual'][symbol]}
                for key,val in fields.items():close(float(saved[key]),val,f'库存{day}/{symbol}/{key}',0)
                require(truth(saved.inventory_preserved)==(diff==0),'库存布尔状态错误')
                require(truth(saved.recovery_failure)==(symbol in active),'恢复失败状态错误')
                require(not truth(saved.weekly_liquidation_failure),'固定底仓错误要求周退出')
                require(stats['actual']['sold'][symbol]<=opened,'每天卖出旧股总额超限')
                if cfg.get('max_cycles_per_day',0):require(cycles[symbol]<=cfg['max_cycles_per_day'],'次数上限超限')
                if diff:inventory_bad.add(day)
                if symbol in active:recovery_bad.add(day)
                raw=self.bars(day.year,symbol)
                for minute,col in [(571,'open'),(900,'close')]:
                    stamp=day+pd.Timedelta(minutes=minute)
                    if stamp in raw.index:marks[symbol]=float(raw.loc[stamp,col])
            saved_curve=curve.iloc[ci]
            for name,book in books.items():
                value=sum(q*marks.get(symbol,0.) for symbol,q in book['qty'].items())
                close(float(saved_curve['cash' if name=='actual' else 'baseline_cash']),book['cash'],'收盘现金')
                close(float(saved_curve['equity' if name=='actual' else 'baseline_equity']),book['cash']+value,'原始价格收盘权益')
                if name=='actual':close(float(saved_curve.holdings_value),value,'收盘持股估值')
            close(float(saved_curve.t_added_equity),float(saved_curve.equity-saved_curve.baseline_equity),'日增量权益')
            close(float(saved_curve.pending_pair_count),len(active),'日末待恢复组',0)
            close(float(saved_curve.completed_pair_count),len(complete),'日末完整组',0)
            same=all(books['actual']['qty'].get(s,0)==books['baseline']['qty'].get(s,0) for s in symbols)
            if same and not active:
                realized=float(trips.loc[trips.second_datetime.le(saved_curve.datetime),'net_pnl'].sum()) if len(trips) else 0.
                close(float(saved_curve.t_added_equity),realized,'无债务同底仓时权益增量=片段净额')
        for part in trips.itertuples(index=False):
            pair,fragment=int(part.pair_id),int(part.fragment_id)
            require(pair in first and (pair,fragment) in recoveries,'片段没有事件')
            one,two=first[pair],recoveries[pair,fragment]
            qty=int(part.shares);require(qty==two.shares,'片段恢复股数和事件不同')
            require(part.first_datetime==one.datetime and part.second_datetime==two.datetime,'配对端点时间错误')
            buy,sell=(one,two) if one.side=='buy' else (two,one)
            close(float(part.buy_price),buy.price,'配对买价',1e-9)
            close(float(part.sell_price),sell.price,'配对卖价',1e-9)
            first_fee=one.fee*qty/one.shares
            buyfee,sellfee=(first_fee,two.fee) if one.side=='buy' else (two.fee,first_fee)
            close(float(part.buy_fee),buyfee,'片段买费')
            close(float(part.sell_fee),sellfee,'片段卖费')
            gross=(sell.price-buy.price)*qty;net=gross-buyfee-sellfee
            close(float(part.gross_pnl),gross,'片段毛收益')
            close(float(part.net_pnl),net,'片段净收益')
            close(float(part.slippage_cost),one.slippage_cost*qty/one.shares+two.slippage_cost,'片段滑点')
            require(truth(part.pair_fully_recovered)==(pair in complete),'片段完整状态错误')
        require(len(trips)==len(recoveries),'恢复事件与片段数不一致')
        grouped=trips.groupby('pair_id').shares.sum().to_dict() if len(trips) else {}
        require(all(grouped.get(pair,0)==first_quantities[pair] for pair in complete),'完整组恢复数量错误')
        position_qty=positions.groupby('symbol').shares.sum().to_dict() if len(positions) else {}
        for symbol,qty in books['actual']['qty'].items():close(position_qty.get(symbol,0),qty,'期末持仓',0)
        baseline_qty=positions.groupby('symbol').baseline_shares.sum().to_dict() if len(positions) else {}
        for symbol,qty in books['baseline']['qty'].items():close(baseline_qty.get(symbol,0),qty,'基准期末持仓',0)
        for position in positions.itertuples(index=False):
            close(float(position.mark_price),marks[position.symbol],'期末原始估值价格',1e-9)
            close(float(position.market_value),position.shares*marks[position.symbol],'期末持股市值')
        require(positions.empty or positions.exit_date.isna().all(),'固定底仓存在自动退出日期')
        require(positions.empty or not positions.overdue_weekly_exit.map(truth).any(),'固定底仓错误标为逾期周退出')
        # 从基础成交键核共同计划，不通过最终数量碰巧相等掩盖差异。
        keys=['datetime','symbol','plan_id']
        abase=base_actual[[*keys,'shares']].copy();bbase=base_events[[*keys,'shares']].copy()
        for part in (abase,bbase):part['plan_id']=part.plan_id.astype(str)
        joined=abase.merge(bbase,on=keys,how='outer',suffixes=('_actual','_baseline'),validate='one_to_one')
        base_mismatch=int(joined.shares_actual.fillna(0).ne(joined.shares_baseline.fillna(0)).sum())
        mismatch_rows=self.csv(folder/'base_plan_mismatches.csv')
        require(len(mismatch_rows)==base_mismatch,'基础计划失配表计数与成交不符')
        expected_failures={'inventory_mismatch_days':len(inventory_bad),'recovery_failure_days':len(recovery_bad),
            'pending_pairs':len(active),'t1_violation_count':0,'base_plan_mismatch_count':base_mismatch}
        for key,value in expected_failures.items():close(float(metrics[key]),value,key,0)
        complete_net=float(trips.loc[trips.pair_id.isin(complete),'net_pnl'].sum()) if len(trips) else 0.
        fragment_net=float(trips.net_pnl.sum()) if len(trips) else 0.
        lots=sum(first_quantities[pair]/cfg['lot_size'] for pair in complete)
        t_events=actual_events.loc[actual_events.event_type.eq('t')]
        expectations={'t_pair_count':len(complete),'completed_pair_count':len(complete),
            'started_pair_count':len(first),'fragment_count':len(trips),'completed_t_pair_lots':lots,
            'total_t_traded_lots':float(t_events.shares.sum()/cfg['lot_size']),
            'all_reference_acquisition_cost':base_cost,'realized_complete_pair_pnl':complete_net,
            'realized_all_fragment_pnl':fragment_net,'t_fee':float(t_events.fee.sum()),
            'incremental_net_pnl':float(curve.equity.iloc[-1]-curve.baseline_equity.iloc[-1])}
        for key,value in expectations.items():close(float(metrics[key]),value,key)
        if base_cost>0:close(float(metrics['holding_cost_reduction_rate']),complete_net/base_cost,'含费原始底仓成本降低率',1e-12)
        else:require(metrics['holding_cost_reduction_rate'] is None,'零成本不能伪造0%')
        output={'path':str(folder.relative_to(self.report)),'passed':True,**expectations,**expected_failures,
                'market_days':len(days),'first_market_day':str(days.min()),'last_market_day':str(days.max()),
                'original_base_cost_includes_fees':True,'no_automatic_base_sale':True}
        self.results.append(output)
        return output

    def run(self):
        """主进程终态后核冻结、所有候选/实际外层及来源，保留内部未达标状态。"""
        completion=self.json(self.report/'runtime_integrity_audit.json')
        require(completion.get('all_unchanged') is True,'主运行缺成功完整终态')
        require((self.report/'final_report.md').is_file(),'主报告尚未完成')
        self.record(self.report/'final_report.md')
        sources=self.json(self.report/'source_hashes.json')
        current_changes={}
        for name,expected in sources.items():
            snapshot=self.report/'source_snapshot'/name
            require(digest(self.record(snapshot))==expected,f'冻结源码快照不符:{name}')
            current=digest(ROOT/name)
            if current!=expected:current_changes[name]={'frozen_sha256':expected,'current_sha256':current}
        reader='quant_backtest/weekly_t_data.py'
        require(digest(ROOT/reader)==sources[reader],'当前复用原CSV读取器与冻结版不同')
        original_inputs=self.json(self.report/'input_sources.json')
        for path,identity in original_inputs.items():
            require(digest(self.record(path))==identity['sha256'],f'原始已登记输入变更:{path}')
        freeze=self.json(self.report/'all_rules_frozen_before_outer_test.json')
        evaluation=self.json(self.report/'outer_evaluation_decision.json')
        goal=self.json(self.report/'goal_evaluation.json')
        cfg=self.cfg;choices=[];candidate_count=0;outer_count=0
        for year in cfg['test_years']:
            folder=self.report/f'year_{year}'
            split=self.json(folder/'split.json')
            require(year not in split['train_years'] and split['inner_year'] not in split['core_years'],'年度隔离错误')
            selected_path=folder/'selected_rule.json';selected=self.json(selected_path)
            require(digest(selected_path)==freeze['selected_rule_sha256'][str(year)],'选中规则偏离全折冻结身份')
            table=self.csv(folder/'validation_rule_metrics.csv')
            require(len(table)==55 and set(table.candidate_index)==set(range(55)),'内部55候选不完整')
            actual=[]
            for index in range(55):
                result=self.account(folder/f'validation_candidates/candidate_{index:03d}')
                row=table.loc[table.candidate_index.eq(index)].iloc[0]
                for key in ['t_pair_count','incremental_net_pnl',*FAILURE_KEYS]:close(float(row[key]),result[key],f'候选汇总{index}/{key}')
                require(int(result['first_market_day'][:4])==split['inner_year'],'内部账户用错年份')
                actual.append({'candidate_index':index,**result});candidate_count+=1
            eligible=[r for r in actual if r['candidate_index']>0 and r['t_pair_count']>0 and all(r[k]==0 for k in FAILURE_KEYS)]
            chosen=sorted(eligible,key=lambda r:(-r['incremental_net_pnl'],r['candidate_index']))[0] if eligible else actual[0]
            require(chosen['candidate_index']==selected['rule']['candidate_index'],'内部选择不是执行合格最高实际净额')
            require(selected_path.stat().st_mtime_ns<=(self.report/'all_rules_frozen_before_outer_test.json').stat().st_mtime_ns,'规则在全折冻结后被写入')
            choices.append({'test_year':year,'candidate_index':chosen['candidate_index'],'inner_net_pnl':chosen['incremental_net_pnl'],
                            'inner_completed_groups':chosen['t_pair_count'],'inner_completed_pair_lots':chosen['completed_t_pair_lots']})
            for scope in ('validation','test'):
                context=folder/f'{scope}_context'
                if (context/'baseline/metrics.json').exists():self.account(context/'baseline')
            if evaluation['outer_evaluation_started']:
                require((folder/'test_equity_curve.csv').stat().st_mtime_ns>=(self.report/'all_rules_frozen_before_outer_test.json').stat().st_mtime_ns,'外层输出在统一冻结前生成')
                result=self.account(folder)
                require(int(result['first_market_day'][:4])==year,'外层账户用错年份')
                outer_count+=1
            print(f'独立账本与原分钟核验完成：fold={year}',flush=True)
        targets=all(self.json(self.report/f'year_{c["test_year"]}/selected_rule.json')['inner_metrics']['holding_cost_reduction_rate']>=cfg['profit_target_rate'] and c['candidate_index']>0 for c in choices)
        require(evaluation['all_inner_cost_targets_met']==targets,'内部5%晋级判定错误')
        expected_outer=not cfg['internal_only'] and (targets or cfg['report_scope']=='smoke')
        require(evaluation['outer_evaluation_started']==expected_outer,'外层晋级政策不符')
        if cfg['report_scope']=='smoke':require(not goal['goal_achieved'],'工程烟测不得宣称完整目标实现')
        changed=[path for path,value in self.inputs.items() if digest(path)!=value]
        require(not changed,f'审计期间输入变更:{changed}')
        return {'passed':True,'complete':True,'candidate_accounts':candidate_count,'outer_accounts':outer_count,
            'all_saved_accounts_checked':len(self.results),'raw_event_checks':self.checked_raw_events,
            'raw_source_files':self.raw_files,'frozen_source_count':len(sources),
            'current_source_changes_after_original_run':current_changes,
            'current_source_changes_are_not_original_snapshot_mutation':True,
            'original_input_count':len(original_inputs),'all_read_inputs_unchanged':True,
            'choices':choices,'accounts':self.results,'outer_evaluation_policy_passed':True,
            'goal_achieved':goal['goal_achieved'],'historical_retest_not_new_untouched_test':True,
            'raw_scope':'全部实际成交参考价/滑点/费用/同原分钟累计容量；持股每日原始开收盘估值。非重新生成全部信号或重放拒单引擎。',
            'same_market_minute_capacity':'09:30开盘映射09:31 raw bar，与同一raw bar其他成交共享容量；两个账户各自独立'}


def main():
    """新建唯一目录归档成功或失败证据，绝不覆盖主报告、源码快照或旧审计。"""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args();report=args.report_dir.resolve()
    require((report/'runtime_integrity_audit.json').is_file(),'主运行尚无terminal成功证据，禁止审计部分结果')
    out=args.output_dir or report/'independent_balanced_audit'/f'run_{datetime.now():%Y%m%d_%H%M%S_%f}'
    out=out.resolve();out.mkdir(parents=True,exist_ok=False)
    shutil.copy2(__file__,out/'audit_balanced_inventory_t.py')
    start=time.monotonic();auditor=None
    try:
        auditor=Auditor(report,out);result=auditor.run()
    except Exception as exc:
        result={'passed':False,'complete':False,'error':str(exc),'traceback':traceback.format_exc()}
    result.update(report_dir=str(report),audit_script_sha256=digest(__file__),
                  started_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-start,
                  inputs_sha256=auditor.inputs if auditor else {})
    (out/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    status='通过' if result['passed'] else '失败'
    text=f'# 固定底仓T独立审计：{status}\n\n报告：{report}\n\n'
    if result['passed']:
        text+=(f"核验{result['candidate_accounts']}个内部候选、{result['outer_accounts']}个外层，"
               f"含底仓规划共{result['all_saved_accounts_checked']}本账户；原CSV成交核对{result['raw_event_checks']}条。\n\n"
               '独立重建每日现金、股份及日初旧股卖额，核原始含费底仓成本、完整组/恢复片段/配对手、固定底仓无周卖出、金额区间和选择冻结顺序。负净额保持负数，交易增加不等于收益改善。\n\n'
               '逐项账户结果、源码快照身份、后续开发导致的当前源码差异及全部输入SHA见 verification.json。原始报告与运行源码快照未修改。\n\n'
               '年度LOYO允许以后年份训练；这是已研究历史的复测，非全新未看过测试。原始行情核查覆盖实际成交及估值，未重新生成信号或模拟所有未成交订单。\n')
    else:text+=result['error']+'\n'
    (out/'review.md').write_text(text,encoding='utf-8')
    print(json.dumps({'passed':result['passed'],'output_dir':str(out),'elapsed_seconds':result['elapsed_seconds'],
                      'error':result.get('error')},ensure_ascii=False),flush=True)
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':
    main()
