"""独立核验v13动态恢复的冻结来源、当前退出意图、原始分钟和两本账户，不调用生产引擎。

必须在主进程结束且runtime_integrity_audit.json落盘后运行。输入--report-dir；
新建独立审计目录，默认不覆盖任何原结果。实际当前源码允许已声明的后续开发，
原运行身份以报告source_snapshot为准；失败也保存脚本、输入SHA和错误信息。
"""
from __future__ import annotations

import argparse
import ast
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
from itertools import product
from types import SimpleNamespace

QUOTE_COLUMNS = ['open','high','low','close','volume','amount']
SESSION = tuple(range(571,691))+tuple(range(781,901))
EXIT_TIMES = ['original_deadline','exit_signal_time','exit_earliest_time']

FAILURE_KEYS = ['inventory_mismatch_days','recovery_failure_days','pending_pairs',
                't1_violation_count','base_plan_mismatch_count']
LEGACY_ENGINE_SHA = '2497e883bc62dd3f468b32c31ceffe7abc76fb84b67a23f447f04317bcb2a467'


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


def update_known_mark(marks, mark_times, symbol, reference, timestamp):
    """独立维护已发生的未滑点报价；较早成交不会覆盖较晚已知价。"""
    require(np.isfinite(reference) and reference>0,'估值参考价必须有限且正')
    if symbol not in mark_times or timestamp>=mark_times[symbol]:
        marks[symbol]=float(reference);mark_times[symbol]=pd.Timestamp(timestamp)


def independent_grid(cfg):
    """独立枚举54入场乘3退出；不调用被审查的策略工厂。"""
    take,stop=cfg['dynamic_take_net_return'],cfg['dynamic_stop_net_return']
    require(take==.003 and stop==.01,'v13退出阈值偏离事前固定值')
    entries=[{'candidate_index':0,'enabled':False,'family':'hold',
              'entry_deviation':0.,'recovery_horizon':'close','fraction':0.}]
    for family,level,horizon,fraction in product(
            ['vwap_reversal','confirmed_reversal','momentum'],[.003,.006,.01],[20,60,'close'],[.5,1.]):
        entries.append(dict(candidate_index=len(entries),enabled=True,family=family,
                            entry_deviation=level,recovery_horizon=horizon,fraction=fraction))
    result=[{**entries[0],'entry_candidate_index':0,'exit_mode':'fixed',
             'take_net_return':take,'stop_net_return':stop}]
    for entry in entries[1:]:
        for mode in ['fixed','profit_buffer','profit_buffer_stop']:
            result.append({**entry,'candidate_index':len(result),'entry_candidate_index':entry['candidate_index'],
                           'exit_mode':mode,'take_net_return':take,'stop_net_return':stop})
    return result


def calculate_first_exit(first, raw, cfg, mode):
    """仅从首腿至原期限前一分钟原close重建首次意图，缺行仍生成时间退出。

    这是独立逐分钟观察计算，不执行账户、不给定未来最佳成交点，也不读取保存
    的退出信号。未锁存之前不可能有恢复片段，因此原数量和原首费均完整保留。
    """
    stamp=pd.Timestamp(first.datetime);deadline=pd.Timestamp(first.planned_recovery_time)
    require(stamp.normalize()==deadline.normalize() and stamp<deadline,'原期限必须同日且晚于首腿')
    legal=pd.DatetimeIndex([stamp.normalize()+pd.Timedelta(minutes=m) for m in SESSION])
    require(stamp in legal and deadline in legal,'首腿或硬期限不在合法会话')
    times=legal[(legal>=stamp)&(legal<deadline)]
    frame=raw.reindex(times)
    norm=float(first.reference_price)*int(first.shares)
    require(norm>0,'原完整首腿参考金额必须为正')
    for time,row in frame.iterrows():
        earliest=next_minute(time)
        vals=[row.get(k,np.nan) for k in ['close','high','low','amount','volume']]
        valid=(all(np.isfinite(vals)) and vals[0]>0 and vals[3]>0 and vals[4]>0
               and vals[2]-1e-6<=vals[0]<=vals[1]+1e-6)
        net,rate=np.nan,np.nan
        if valid:
            reverse='sell' if first.side=='buy' else 'buy'
            price=float(row.close)*(1+cfg['slippage_rate']*(1 if reverse=='buy' else -1))
            gross=int(first.shares)*(first.price-price if reverse=='buy' else price-first.price)
            net=gross-fee(int(first.shares)*price,reverse,time,cfg)-first.fee
            rate=net/norm
        reason=''
        if earliest>=deadline:reason='hard_deadline'
        elif mode!='fixed' and valid:
            if rate>=cfg['dynamic_take_net_return']:reason='profit_buffer'
            elif mode=='profit_buffer_stop' and rate<=-cfg['dynamic_stop_net_return']:reason='stop_loss'
        if reason:
            return {'exit_signal_time':time,'exit_earliest_time':earliest,'exit_reason':reason,
                    'observed_price':float(row.close) if valid else np.nan,
                    'estimated_net_pnl':net,'estimated_net_return':rate,
                    'normalization_notional':norm,'remaining_shares':int(first.shares),
                    'realized_fragment_net_pnl':0.}
    raise AssertionError('合法硬期限没有产生必需退出意图')


def compare_exit(saved, expected):
    """逐字段核首次退出；金额只用财务容差，时间及触发选择完全相等。"""
    for name,value in expected.items():
        actual=getattr(saved,name)
        if isinstance(value,pd.Timestamp) or isinstance(value,str):
            require(actual==value,f'退出{name}与独立首次触发不同:{actual}/{value}')
        elif np.isnan(value):require(pd.isna(actual),f'无有效观察时{name}不能伪造已知值')
        else:close(float(actual),value,f'退出{name}',1e-12 if name=='estimated_net_return' else 1e-6)


class Auditor:
    """持有只读文件身份和原CSV缓存；账户账簿完全由本类独立计算。"""

    def __init__(self,report,out):
        self.report,self.out=report,out
        self.inputs={};self.raw={};self.full_raw={};self.raw_files={};self.results=[]
        self.cfg=self.json(report/'resolved_config.json')
        self.protocol=self.json(report/'research_protocol.json')
        require(self.protocol['valuation_contract']=='full_minute_reference_stream_then_daily_close','未知动态估值合同')
        require(self.cfg['research_version']=='dynamic_inventory_t_v13','本审计仅支持v13')
        market_path=Path(self.cfg['market_cache_dir'])/'daily.pkl'
        if not market_path.is_absolute():market_path=ROOT/market_path
        market=pd.read_pickle(self.record(market_path))
        self.market_calendar=pd.DatetimeIndex(parse_time(market.date).unique()).normalize().sort_values()
        self.checked_raw_events=0;self.checked_exit_pairs=0;self.cache_audits=[]
        self.sparse_arrays={};self.mark_cache={};self.exit_cache={};self.base_anchors={}
        self.valuation_mode='full_minute_reference_stream_then_daily_close'

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
        """独立读取原CSV；完整真实行与旧日端点合法行分别保存，不导入生产读取器。"""
        key=(int(year),str(symbol))
        if key not in self.raw:
            path=Path(self.cfg['data_root'])/str(year)/f'{symbol}.csv'
            if not path.is_absolute():path=ROOT/path
            self.record(path)
            columns=['datetime',*QUOTE_COLUMNS]
            frame=pd.read_csv(path,usecols=columns)
            frame['datetime']=parse_time(frame.datetime)
            frame=frame.loc[frame.datetime.dt.year.eq(year)].copy()
            require(not frame.datetime.duplicated().any(),'原始时间重复，不能任选一条')
            frame[QUOTE_COLUMNS]=frame[QUOTE_COLUMNS].apply(pd.to_numeric,errors='raise')
            minutes=frame.datetime.dt.hour*60+frame.datetime.dt.minute
            session=minutes.isin(SESSION)
            require((session|minutes.eq(570)).all(),'原始分钟有非法会话时钟')
            frame=frame.loc[session].sort_values('datetime').set_index('datetime')
            require(not np.isinf(frame[QUOTE_COLUMNS].to_numpy()).any(),'原始存在无穷值')
            require(not frame[QUOTE_COLUMNS].lt(0).any().any(),'原始负价量')
            valid=np.isfinite(frame[QUOTE_COLUMNS]).all(axis=1)
            valid&=frame[['open','high','low','close']].gt(0).all(axis=1)
            valid&=frame.high.add(1e-6).ge(frame[['open','close']].max(axis=1))
            valid&=frame.low.sub(1e-6).le(frame[['open','close']].min(axis=1))
            valid&=frame.high.ge(frame.low)
            self.full_raw[key]=frame
            self.raw[key]=frame.loc[valid]
            self.raw_files[str(path.resolve())]={'sha256':self.inputs[str(path.resolve())],
                'full_session_rows':len(frame),'old_endpoint_valid_rows':int(valid.sum())}
        return self.raw[key]

    def full_bars(self,year,symbol):
        """保留NaN/零成交完整raw；观察可用性与成交资格由独立核查分别判断。"""
        self.bars(year,symbol)
        return self.full_raw[int(year),str(symbol)]

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

    def dense_arrays(self,path):
        """独立解析npy与JSON格式并核全部数组SHA，不导入生产store类。"""
        path=Path(path).resolve();metadata=self.json(path/'metadata.json')
        identity=metadata['identity']
        encoded=json.dumps(identity,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()
        require(hashlib.sha256(encoded).hexdigest()==metadata['fingerprint'],'dense身份摘要不符')
        require(path.name=='dense_recovery_'+metadata['fingerprint'],'dense目录名与身份不符')
        require(identity['quote_columns']==QUOTE_COLUMNS,'dense列顺序不符')
        symbols=metadata['symbols'];require(symbols==sorted(set(symbols)),'dense股票键非法')
        arrays={}
        for name,record in metadata['arrays'].items():
            require(name in {'times.npy','quotes.npy','present.npy'},'dense未知数组')
            target=self.record(path/name)
            require(self.inputs[str(target)]==record['sha256'],'dense数组字节SHA不符')
            arrays[name]=np.load(target,mmap_mode='r',allow_pickle=False)
        times=arrays['times.npy'];present=arrays['present.npy'];quotes=arrays['quotes.npy']
        require(times.dtype==np.dtype('int64') and present.dtype==np.dtype('uint8') and quotes.dtype==np.dtype('float64'),'dense dtype不符')
        require(present.shape==(len(times),len(symbols)) and quotes.shape==(*present.shape,6),'dense形状不符')
        require(len(times)==metadata['time_count'] and metadata['time_count']==identity['time_count'],'dense时间计数不同')
        require(not len(times) or (np.diff(times)>0).all(),'dense时间键未严格递增')
        require(np.isin(present,[0,1]).all(),'dense present掩码非法')
        return metadata,times,present,quotes

    def audit_context(self,context,year):
        """逐实际键六列绑定raw到完整mmap，核全日历、不补缺行和冻结基础成交。"""
        audit=self.json(context/'full_minute_coverage_audit.json')
        manifest=self.json(audit['manifest_path'])
        identity=manifest['identity'];names=identity['symbols']
        encoded=json.dumps(identity,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()
        require(hashlib.sha256(encoded).hexdigest()==manifest['request_fingerprint']==audit['request_fingerprint'],'full请求摘要不符')
        require(identity['year']==year,'full请求年度错误')
        days=self.market_calendar[self.market_calendar.year==year]
        if self.cfg.get('smoke_days',0):days=days[:self.cfg['smoke_days']]
        require(identity['market_days']==[str(d.date()) for d in days],'full请求遗漏真实市场日/现金期')
        metadata,times,present,quotes=self.dense_arrays(manifest['dense_cache_path'])
        require(digest(Path(manifest['dense_cache_path'])/'metadata.json')==manifest['dense_metadata_sha256'],'full与dense身份不符')
        require(digest(self.report/'source_snapshot/quant_backtest/dynamic_t_data.py')==identity['reader_sha256'],'full raw读取代码未绑定运行快照')
        pieces=[];per_symbol=[]
        for symbol in names:
            frame=self.full_bars(year,symbol)
            frame=frame.loc[frame.index.normalize().isin(days)].reset_index().assign(symbol=symbol)
            frame=frame[['datetime','symbol',*QUOTE_COLUMNS]]
            frame[QUOTE_COLUMNS]=frame[QUOTE_COLUMNS].astype('float64')
            pieces.append(frame)
            raw_path=(Path(self.cfg['data_root'])/str(year)/f'{symbol}.csv').resolve()
            require(identity['raw_sources'][str(raw_path)]['sha256']==self.inputs[str(raw_path)],'full原CSV绑定错误')
            per_symbol.append({'symbol':symbol,'present_rows':len(frame),'missing_rows':len(days)*240-len(frame)})
        frame=pd.concat(pieces,ignore_index=True) if pieces else pd.DataFrame(columns=['datetime','symbol',*QUOTE_COLUMNS])
        frame=frame.sort_values(['datetime','symbol']).reset_index(drop=True)
        declared=metadata['identity']
        require(len(frame)==declared['row_count']==int(np.count_nonzero(present)),'full实际行数不符/填了缺失行')
        require(metadata['symbols']==sorted(frame.symbol.unique()),'full股票维度与raw不同')
        require(np.array_equal(times,pd.DatetimeIndex(frame.datetime.unique()).asi8),'full时间维度与raw不同')
        input_sha=hashlib.sha256(pd.util.hash_pandas_object(frame,index=False).to_numpy(dtype='<u8').tobytes()).hexdigest()
        require(input_sha==declared['input_sha256'],'full规范化raw输入SHA不符')
        ti=pd.Index(times).get_indexer(pd.DatetimeIndex(frame.datetime).asi8)
        si=pd.Index(metadata['symbols']).get_indexer(frame.symbol)
        require((ti>=0).all() and (si>=0).all() and present[ti,si].all(),'full遗漏原始实际键')
        require(np.array_equal(frame[QUOTE_COLUMNS].to_numpy(float),quotes[ti,si],equal_nan=True),'full真实六列量价不符')
        require(audit['symbols']==names and audit['market_days']==len(days),'full审计日历/股票不符')
        require(audit['present_rows']==len(frame) and audit['missing_rows']==len(days)*240*len(names)-len(frame),'full缺失审计不符')
        for row,expected in zip(audit['per_symbol'],per_symbol):
            require(all(row[k]==v for k,v in expected.items()),'full逐股缺失计数错误')
        alignment=self.json(context/'execution_quote_alignment_audit.json')
        require(alignment.get('no_signal_rows_filtered') is True,'quote核对被用作未来删信号')
        require(alignment.get('all_declared_execution_values_match_raw',alignment['verified_execution_rows']==0) is True,'首腿原quote核对未通过')
        frozen=self.csv(context/'frozen_base_selections.csv')
        require(set(frozen.symbol)==set(names),'全分钟股票并集偏离因果冻结持股')
        plan=self.json(context/'base_plan_identity.json')
        require(digest(context/'frozen_base_selections.csv')==plan['frozen_csv_sha256'],'底仓冻结CSV身份错误')
        basis=self.csv(context/'frozen_base_entries.csv')
        baseline=self.csv(context/'baseline/baseline_trade_events.csv')
        planning=self.csv(context/'planning_baseline/baseline_trade_events.csv')
        keys=['datetime','symbol','side','shares','reference_price','price','notional','fee','base_order_type']
        def values(frame):
            frame=frame[keys].copy();frame['datetime']=parse_time(frame.datetime)
            return frame.sort_values(['datetime','symbol'],kind='stable').reset_index(drop=True)
        pd.testing.assert_frame_equal(values(basis),values(planning),check_exact=True)
        pd.testing.assert_frame_equal(values(basis),values(baseline),check_exact=True)
        self.base_anchors[str(context.resolve())]=values(baseline)
        self.cache_audits.append({'context':str(context.relative_to(self.report)),
            'year':year,'raw_rows':len(frame),'symbols':len(names),'market_days':len(days),
            'missing_rows':audit['missing_rows'],'raw_input_sha256':input_sha,
            'all_raw_keys_six_columns_exact':True,'frozen_base_trades_equal_planning_and_final':True})

    def update_closing_mark(self,day,symbol,planning,marks,mark_times):
        """分别核完整行情与稀疏规划的最后已知价；只用原始未滑点close。

        精确有效日收盘优先，因此一般无需打开旧大稀疏缓存。缺日收盘时规划只
        可看到原store确实覆盖的时钟及人工当前quote，不能借新完整缓存补价。
        """
        key=(day,symbol,planning)
        if key not in self.mark_cache:
            end=day+pd.Timedelta(minutes=900);old=self.bars(day.year,symbol)
            if end in old.index:
                latest=(end,float(old.loc[end,'close']))
            elif not planning:
                raw=self.full_bars(day.year,symbol)
                subset=raw.loc[raw.index.normalize().eq(day) if hasattr(raw.index.normalize(),'eq') else raw.index.normalize()==day]
                subset=subset.loc[np.isfinite(subset.close)&subset.close.gt(0)]
                latest=(subset.index[-1],float(subset.close.iloc[-1])) if len(subset) else None
            else:
                if day.year not in self.sparse_arrays:
                    self.sparse_arrays[day.year]=self.dense_arrays(self.cfg['dense_recovery_cache_paths'][str(day.year)])
                meta,times,present,quotes=self.sparse_arrays[day.year]
                latest=None
                if symbol in meta['symbols']:
                    column=meta['symbols'].index(symbol)
                    indices=np.flatnonzero((times>=day.value)&(times<=end.value))
                    indices=indices[present[indices,column].astype(bool)]
                    indices=indices[np.isfinite(quotes[indices,column,3])&(quotes[indices,column,3]>0)]
                    if len(indices):
                        last=indices[-1];stamp=pd.Timestamp(times[last]);price=float(quotes[last,column,3])
                        raw=self.full_bars(day.year,symbol)
                        require(stamp in raw.index,'规划稀疏quote在原CSV缺失')
                        close(price,float(raw.loc[stamp,'close']),'规划稀疏估值与raw',1e-10)
                        latest=(stamp,price)
                if self.cfg.get('manual_buys_csv'):
                    manual=self.csv(self.cfg['manual_buys_csv']);manual['datetime']=parse_time(manual.datetime)
                    manual=manual.loc[manual.symbol.eq(symbol)&manual.datetime.dt.normalize().eq(day)]
                    raw=self.full_bars(day.year,symbol)
                    for stamp in manual.datetime:
                        if stamp in raw.index and np.isfinite(raw.loc[stamp,'close']) and raw.loc[stamp,'close']>0:
                            if latest is None or stamp>latest[0]:latest=(stamp,float(raw.loc[stamp,'close']))
            self.mark_cache[key]=latest
        latest=self.mark_cache[key]
        if latest is not None:update_known_mark(marks,mark_times,symbol,latest[1],latest[0])

    def audit_exits(self,folder,events,trips,positions,metrics,days):
        """逐组重建首次触发并核所有后续恢复/拒单保留原锁存意图及每分钟机会。"""
        saved=self.csv(folder/'exit_decisions.csv')
        for key in ['first_signal_time','first_execution_time',*EXIT_TIMES]:
            if key in saved:saved[key]=parse_time(saved[key])
        require(not saved.pair_id.duplicated().any(),'同组退出意图重复/被分片重复计数')
        first=events.loc[events.event_type.eq('t')&events.fragment_id.eq(0)]
        require(set(saved.pair_id)==set(first.pair_id),'首腿缺少唯一硬期限/动态退出意图')
        mode=metrics['dynamic_exit_mode']
        if (folder/'rule.json').exists():require(mode==self.json(folder/'rule.json')['exit_mode'],'实际模式偏离候选规则')
        elif (folder/'selected_rule.json').exists():require(mode==self.json(folder/'selected_rule.json')['rule']['exit_mode'],'外层模式偏离冻结规则')
        else:require(mode=='fixed' and first.empty,'规划/基准不应开启T')
        close(metrics['dynamic_take_net_return'],self.cfg['dynamic_take_net_return'],'冻结止盈阈值',0)
        close(metrics['dynamic_stop_net_return'],self.cfg['dynamic_stop_net_return'],'冻结止损阈值',0)
        blocked=self.csv(folder/'blocked_orders.csv')
        for key in ['datetime','signal_datetime']:
            if key in blocked:blocked[key]=parse_time(blocked[key])
        attempts=blocked.loc[blocked.event_type.eq('t_recovery')].copy()
        observed=set()
        for one in first.itertuples(index=False):
            intent=saved.loc[saved.pair_id.eq(one.pair_id)].iloc[0]
            require(intent.first_signal_time==one.signal_datetime and intent.first_execution_time==one.datetime,'退出原始首腿身份错')
            require(intent.original_deadline==one.planned_recovery_time,'退出硬期限被改写')
            require(intent.symbol==one.symbol and intent.direction==('buy_first' if one.side=='buy' else 'sell_first'),'退出股票或方向错')
            require(intent.exit_mode==mode,'退出意图模式错')
            key=(one.datetime,one.symbol,one.shares,one.side,one.price,one.fee,one.planned_recovery_time,mode)
            if key not in self.exit_cache:
                self.exit_cache[key]=calculate_first_exit(one,self.full_bars(one.datetime.year,one.symbol),self.cfg,mode)
            expected=self.exit_cache[key];compare_exit(intent,expected)
            fragments=events.loc[events.event_type.eq('t')&events.pair_id.eq(one.pair_id)&events.fragment_id.gt(0)].copy()
            for field in EXIT_TIMES:fragments[field]=parse_time(fragments[field])
            for part in fragments.itertuples(index=False):
                require(part.datetime>=intent.exit_earliest_time and part.datetime>intent.exit_signal_time,'恢复提前或同bar成交')
                require(part.signal_datetime==intent.exit_signal_time,'锁存后恢复信号被改写')
                for field in [*EXIT_TIMES,'exit_reason']:
                    require(getattr(part,field)==intent[field],'恢复片段更改原退出意图')
                require(part.planned_recovery_time==intent.original_deadline,'片段原计划期限被改写')
            rows=attempts.loc[attempts.symbol.eq(one.symbol)&attempts.signal_datetime.eq(intent.exit_signal_time)]
            require(rows.empty or rows.account.eq('actual').all(),'恢复拒单账户错误')
            stop=fragments.datetime.max() if fragments.shares.sum()==one.shares else days[-1]+pd.Timedelta(minutes=900)
            clock=set()
            for day in days[(days>=one.datetime.normalize())&(days<=stop.normalize())]:
                for minute in (570,*SESSION):
                    stamp=day+pd.Timedelta(minutes=minute)
                    if intent.exit_earliest_time<=stamp<=stop:clock.add(stamp)
            actual=set(rows.datetime)|set(fragments.datetime)
            require(actual==clock,'锁存退出没有保留逐分钟尝试，或出现不合法额外时刻')
            observed.update(rows.index)
            for part in trips.loc[trips.pair_id.eq(one.pair_id)].itertuples(index=False):
                for field in EXIT_TIMES:require(parse_time(getattr(part,field))==intent[field],'配对表退出时间不符')
                require(part.exit_reason==intent.exit_reason,'配对表退出原因不符')
        require(observed==set(attempts.index),'有恢复拒单缺少已锁存首腿')
        self.checked_exit_pairs+=len(first)
        for key,value in {'exit_decision_count':len(saved),
            'profit_exit_decision_count':int(saved.exit_reason.eq('profit_buffer').sum()),
            'stop_exit_decision_count':int(saved.exit_reason.eq('stop_loss').sum()),
            'deadline_exit_decision_count':int(saved.exit_reason.eq('hard_deadline').sum())}.items():close(metrics[key],value,key,0)
        return {'raw_first_exit_intents':len(first),'first_latch_and_all_retry_times_verified':True}

    def account(self,folder):
        """逐日从两本成交事件重建现金、旧股、配对与原始底仓成本并对账。"""
        cfg=self.cfg
        planning=folder.name=='planning_baseline'
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
        if folder.name in ('planning_baseline','baseline'):context=folder.parent
        elif folder.parent.name=='validation_candidates':context=folder.parent.parent/'validation_context'
        else:context=folder/'test_context'
        anchor=self.base_anchors.get(str(context.resolve()))
        if anchor is not None:
            actual_base=base_events[anchor.columns].sort_values(['datetime','symbol'],kind='stable').reset_index(drop=True)
            pd.testing.assert_frame_equal(actual_base,anchor,check_exact=True)
        base_actual=actual_events.loc[actual_events.event_type.ne('t')]
        require(base_events.empty or base_events.side.eq('buy').all(),'基准出现自动或人工底仓卖出')
        require(base_actual.empty or base_actual.side.eq('buy').all(),'实际出现底仓卖出')
        base_cost=float((base_events.notional+base_events.fee).sum())
        auto=base_events.loc[base_events.base_order_type.eq('automatic')]
        require(auto.symbol.nunique()==len(auto),'自动底仓同股不止首次成功买入')
        require(auto.notional.between(cfg['min_position_notional']-1e-7,cfg['max_position_notional']+1e-7).all(),'自动底仓名义金额超区间')
        auto_actual=base_actual.loc[base_actual.base_order_type.eq('automatic')]
        require(auto_actual.notional.between(cfg['min_position_notional']-1e-7,cfg['max_position_notional']+1e-7).all(),'实际自动底仓金额超区间')
        marks={};mark_times={};first={};recoveries={};active={};complete=set()
        inventory_bad=set();recovery_bad=set();base_mismatch=0
        complete_at={};first_quantities={}
        for ci,day in enumerate(days):
            opening={name:dict(book['qty']) for name,book in books.items()}
            day_symbols=set(books['actual']['qty'])|set(books['baseline']['qty'])
            for book in books.values():
                todays=book['groups'].get(day,pd.DataFrame())
                if len(todays):day_symbols.update(todays.symbol)
            for symbol in day_symbols:
                raw=self.bars(day.year,symbol);stamp=day+pd.Timedelta(minutes=571)
                if stamp in raw.index:
                    update_known_mark(marks,mark_times,symbol,float(raw.loc[stamp,'open']),day+pd.Timedelta(minutes=570))
            stats={name:{'sold':defaultdict(int),'base_buys':defaultdict(int),'manual':defaultdict(int)} for name in books}
            cycles=defaultdict(int)
            for name,book in books.items():
                old=defaultdict(int,opening[name])
                rows=book['groups'].get(day,pd.DataFrame())
                for event in rows.itertuples(index=False):
                    self.verify_event(event,book['used'])
                    if event.event_type!='t':
                        update_known_mark(marks,mark_times,event.symbol,float(event.reference_price),event.datetime)
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
                            require(event.side!=original.side and event.datetime>original.datetime,'恢复方向或时间错误')
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
                self.update_closing_mark(day,symbol,planning,marks,mark_times)
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
        exit_audit=self.audit_exits(folder,actual_events,trips,positions,metrics,days)
        output={'path':str(folder.relative_to(self.report)),'passed':True,**expectations,**expected_failures,
                'market_days':len(days),'first_market_day':str(days.min()),'last_market_day':str(days.max()),
                'original_base_cost_includes_fees':True,'no_automatic_base_sale':True,
                'valuation_scope':'sparse_planning_not_final_baseline' if planning else 'common_full_minute',**exit_audit}
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
        require(len(sources)==43,'v13冻结运行源码数量应为43')
        require(completion['runtime_sources_verified']==43,'主程序源码计数不符')
        require(set(sources).issuperset({'quant_backtest/dynamic_t_engine.py','quant_backtest/dynamic_t_data.py','scripts/run_dynamic_inventory_t.py'}),'动态运行源码缺失')
        original_inputs=self.json(self.report/'input_sources.json')
        for path,identity in original_inputs.items():
            require(digest(self.record(path))==identity['sha256'],f'原始已登记输入变更:{path}')
        freeze=self.json(self.report/'all_rules_frozen_before_outer_test.json')
        evaluation=self.json(self.report/'outer_evaluation_decision.json')
        goal=self.json(self.report/'goal_evaluation.json')
        cfg=self.cfg;choices=[];candidate_count=0;outer_count=0
        rules=independent_grid(cfg)
        require(self.protocol['t_rules']==rules,'预声明163规则与独立枚举不一致')
        for year in cfg['test_years']:
            folder=self.report/f'year_{year}'
            split=self.json(folder/'split.json')
            require(year not in split['train_years'] and split['inner_year'] not in split['core_years'],'年度隔离错误')
            selected_path=folder/'selected_rule.json';selected=self.json(selected_path)
            require(digest(selected_path)==freeze['selected_rule_sha256'][str(year)],'选中规则偏离全折冻结身份')
            table=self.csv(folder/'validation_rule_metrics.csv')
            require(len(table)==163 and set(table.candidate_index)==set(range(163)),'内部163候选不完整')
            self.audit_context(folder/'validation_context',split['inner_year'])
            if evaluation['outer_evaluation_started']:self.audit_context(folder/'test_context',year)
            actual=[]
            for index in range(163):
                rule=self.json(folder/f'validation_candidates/candidate_{index:03d}/rule.json')
                require(rule==rules[index],'实际候选规则与冻结预声明不符')
                result=self.account(folder/f'validation_candidates/candidate_{index:03d}')
                row=table.loc[table.candidate_index.eq(index)].iloc[0]
                for key in ['t_pair_count','incremental_net_pnl',*FAILURE_KEYS]:close(float(row[key]),result[key],f'候选汇总{index}/{key}')
                require(int(result['first_market_day'][:4])==split['inner_year'],'内部账户用错年份')
                actual.append({'candidate_index':index,**result});candidate_count+=1
            eligible=[r for r in actual if r['candidate_index']>0 and r['t_pair_count']>0 and all(r[k]==0 for k in FAILURE_KEYS)]
            chosen=sorted(eligible,key=lambda r:(-r['incremental_net_pnl'],r['candidate_index']))[0] if eligible else actual[0]
            require(chosen['candidate_index']==selected['rule']['candidate_index'],'内部选择不是执行合格最高实际净额')
            require(selected['rule']==rules[chosen['candidate_index']],'选中规则参数被改写')
            require(selected_path.stat().st_mtime_ns<=(self.report/'all_rules_frozen_before_outer_test.json').stat().st_mtime_ns,'规则在全折冻结后被写入')
            choices.append({'test_year':year,'candidate_index':chosen['candidate_index'],'inner_net_pnl':chosen['incremental_net_pnl'],
                            'inner_completed_groups':chosen['t_pair_count'],'inner_completed_pair_lots':chosen['completed_t_pair_lots']})
            for scope in ('validation','test'):
                context=folder/f'{scope}_context'
                for account_name in ('planning_baseline','baseline'):
                    if (context/account_name/'metrics.json').exists():self.account(context/account_name)
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
            'full_minute_cache_audits':self.cache_audits,'raw_exit_intent_checks':self.checked_exit_pairs,
            'current_source_changes_after_original_run':current_changes,
            'current_source_changes_are_not_original_snapshot_mutation':True,
            'original_input_count':len(original_inputs),'all_read_inputs_unchanged':True,
            'valuation_contract':self.valuation_mode,
            'choices':choices,'accounts':self.results,'outer_evaluation_policy_passed':True,
            'goal_achieved':goal['goal_achieved'],'historical_retest_not_new_untouched_test':True,
            'raw_scope':'全部实际成交/费用/同原分钟累计容量；完整raw到dense逐键逐列；所有首腿从原始当前close逐分钟重建首次退出触发；独立日终两本账户估值。未重新生成所有首腿信号或重算未成交订单的容量/限价原因。',
            'planning_valuation_scope':'规划使用旧稀疏报价加人工当前分钟；最终基准及候选共用完整raw分钟，二者缺收盘估值允许不同且分别核算',
            'same_market_minute_capacity':'09:30开盘映射09:31 raw bar，与同一raw bar其他成交共享容量；两个账户各自独立'}


def run_self_tests():
    """内置独立反例仅运行合成量价和纯核验函数，不调用生产引擎或覆盖报告。"""
    import unittest
    class Cases(unittest.TestCase):
        def setUp(self):
            self.cfg={'commission_rate':0.,'min_commission':0.,'transfer_rate':0.,
                'stamp_rate_before_20230828':0.,'stamp_rate_from_20230828':0.,'slippage_rate':0.,
                'dynamic_take_net_return':.003,'dynamic_stop_net_return':.01,'lot_size':100,
                'max_bar_amount_fraction':.05}
            self.first=SimpleNamespace(datetime=pd.Timestamp('2025-01-06 10:01'),
                planned_recovery_time=pd.Timestamp('2025-01-06 10:21'),reference_price=10.,
                price=10.,shares=300,side='sell',fee=0.)
            times=pd.DatetimeIndex([pd.Timestamp('2025-01-06')+pd.Timedelta(minutes=m) for m in SESSION])
            self.raw=pd.DataFrame({'close':10.,'open':10.,'high':10.,'low':10.,'amount':20000.,'volume':2000.},index=times)
        def changed(self,clock,price,**fields):
            stamp=pd.Timestamp('2025-01-06 '+clock)
            self.raw.loc[stamp,['open','high','low','close']]=price
            for k,v in fields.items():self.raw.loc[stamp,k]=v
        def value(self,mode='profit_buffer'):
            return calculate_first_exit(self.first,self.raw,self.cfg,mode)
        def test_current_close_then_next_minute(self):
            self.changed('10:02',9.9);v=self.value()
            self.assertEqual(v['exit_signal_time'],pd.Timestamp('2025-01-06 10:02'))
            self.assertEqual(v['exit_earliest_time'],pd.Timestamp('2025-01-06 10:03'))
            self.assertEqual(v['estimated_net_pnl'],(10.-9.9)*300)
        def test_future_prices_cannot_change_first_intent(self):
            self.changed('10:02',9.9);a=self.value();self.changed('10:03',12.)
            self.assertEqual(a,self.value())
        def test_intrabar_extreme_is_not_a_trigger(self):
            self.raw.loc[pd.Timestamp('2025-01-06 10:02'),'low']=9.
            self.assertEqual(self.value()['exit_reason'],'hard_deadline')
        def test_missing_hard_deadline_observation_still_latches(self):
            self.raw=self.raw.drop(pd.Timestamp('2025-01-06 10:20'))
            v=self.value();self.assertTrue(np.isnan(v['observed_price']))
            self.assertEqual(v['exit_earliest_time'],self.first.planned_recovery_time)
        def test_lunch_predecessor_is_1130(self):
            self.first.datetime=pd.Timestamp('2025-01-06 11:29')
            self.first.planned_recovery_time=pd.Timestamp('2025-01-06 13:01')
            self.assertEqual(self.value()['exit_signal_time'],pd.Timestamp('2025-01-06 11:30'))
        def test_zero_volume_cannot_create_profit_observation(self):
            self.changed('10:02',9.9,volume=0.)
            self.assertEqual(self.value()['exit_reason'],'hard_deadline')
        def test_minimum_both_leg_fees_and_original_denominator(self):
            self.cfg['min_commission']=5.;self.first.fee=5.
            self.changed('10:02',9.95)
            self.assertEqual(self.value()['exit_reason'],'hard_deadline')
            self.changed('10:02',9.9);v=self.value()
            self.assertAlmostEqual(v['estimated_net_pnl'],20.)
            self.assertEqual(v['normalization_notional'],3000.)
            self.assertAlmostEqual(v['estimated_net_return'],20/3000)
        def test_fixed_mode_does_not_take_early_profit(self):
            self.changed('10:02',9.9)
            self.assertEqual(self.value('fixed')['exit_reason'],'hard_deadline')
        def test_stop_is_directional_and_mode_specific(self):
            self.first.side='buy';self.changed('10:02',9.8)
            self.assertEqual(self.value('profit_buffer_stop')['exit_reason'],'stop_loss')
            self.assertEqual(self.value()['exit_reason'],'hard_deadline')
        def test_changed_saved_latch_is_rejected(self):
            self.changed('10:02',9.9);v=self.value();s=SimpleNamespace(**v)
            compare_exit(s,v);s.exit_signal_time+=pd.Timedelta(minutes=1)
            with self.assertRaises(AssertionError):compare_exit(s,v)
        def test_open_and_0931_capacity_are_one_raw_bar(self):
            auditor=Auditor.__new__(Auditor);auditor.cfg=self.cfg;auditor.checked_raw_events=0
            rawtime=pd.Timestamp('2025-01-06 09:31')
            auditor.raw_quote=lambda stamp,symbol:(rawtime,10.,20000.)
            used=defaultdict(float)
            one=SimpleNamespace(shares=100,datetime=rawtime-pd.Timedelta(minutes=1),side='sell',symbol='A',
                reference_price=10.,price=10.,notional=1000.,fee=0.,slippage_cost=0.,signal_datetime=pd.Timestamp('2025-01-03 15:00'))
            auditor.verify_event(one,used);one.datetime=rawtime
            with self.assertRaises(AssertionError):auditor.verify_event(one,used)
        def test_complete_quote_mark_survives_missing_daily_close(self):
            auditor=Auditor.__new__(Auditor);auditor.mark_cache={}
            raw=self.raw.loc[:pd.Timestamp('2025-01-06 14:59')].copy();raw.iloc[-1,raw.columns.get_loc('close')]=10.25
            auditor.bars=lambda year,symbol:raw
            auditor.full_bars=lambda year,symbol:raw
            marks={'A':10.};times={'A':pd.Timestamp('2025-01-06 09:30')}
            auditor.update_closing_mark(pd.Timestamp('2025-01-06'),'A',False,marks,times)
            self.assertEqual(marks['A'],10.25)
            self.assertEqual(times['A'],pd.Timestamp('2025-01-06 14:59'))
        def test_later_valid_daily_close_overrides_earlier_reference(self):
            marks={'A':8.};times={'A':pd.Timestamp('2025-01-06 15:00')}
            update_known_mark(marks,times,'A',15.,pd.Timestamp('2025-01-06 14:59'))
            self.assertEqual(marks['A'],8.)
        def test_finite_crossing_fee_date(self):
            self.cfg.update(stamp_rate_before_20230828=.001,stamp_rate_from_20230828=.0005)
            self.assertEqual(fee(10000,'sell',pd.Timestamp('2023-08-27'),self.cfg),10.)
            self.assertEqual(fee(10000,'sell',pd.Timestamp('2023-08-28'),self.cfg),5.)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Cases))
    if not result.wasSuccessful():raise SystemExit(1)


def main():
    """新建唯一目录归档成功或失败证据，绝不覆盖主报告、源码快照或旧审计。"""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir',type=Path)
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args()
    if args.self_test:
        run_self_tests();return
    require(args.report_dir is not None,'请指定--report-dir')
    report=args.report_dir.resolve()
    require((report/'runtime_integrity_audit.json').is_file(),'主运行尚无terminal成功证据，禁止审计部分结果')
    out=args.output_dir or report/'independent_dynamic_audit'/f'run_{datetime.now():%Y%m%d_%H%M%S_%f}'
    out=out.resolve();out.mkdir(parents=True,exist_ok=False)
    shutil.copy2(__file__,out/'audit_dynamic_inventory_t.py')
    start=time.monotonic();started_utc=datetime.now(timezone.utc).isoformat();auditor=None
    try:
        auditor=Auditor(report,out);result=auditor.run()
    except Exception as exc:
        result={'passed':False,'complete':False,'error':str(exc),'traceback':traceback.format_exc()}
    result.update(report_dir=str(report),audit_script_sha256=digest(__file__),
                  started_utc=started_utc,elapsed_seconds=time.monotonic()-start,
                  inputs_sha256=auditor.inputs if auditor else {})
    (out/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    status='通过' if result['passed'] else '失败'
    text=f'# 动态恢复固定底仓T独立审计：{status}\n\n报告：{report}\n\n'
    if result['passed']:
        text+=(f"核验{result['candidate_accounts']}个内部候选、{result['outer_accounts']}个外层，"
               f"含底仓规划共{result['all_saved_accounts_checked']}本账户；原CSV成交核对{result['raw_event_checks']}条。\n\n"
               '独立重建每日现金、股份及日初旧股卖额，核原始含费底仓成本、完整组/恢复片段/配对手、固定底仓无周卖出、金额区间和选择冻结顺序。负净额保持负数，交易增加不等于收益改善。\n\n'
               '额外核所有组的首次退出触发、下一合法分钟延迟、不可撤销的逐分钟尝试和共同完整行情估值；规划稀疏行情估值另列。\n\n'
               '逐项账户结果、源码快照身份、后续开发导致的当前源码差异及全部输入SHA见 verification.json。原始报告与运行源码快照未修改。\n\n'
               '年度LOYO允许以后年份训练；这是已研究历史的复测，非全新未看过测试。原始行情核查覆盖实际成交及估值，未重新生成信号或模拟所有未成交订单。\n')
    else:text+=result['error']+'\n'
    (out/'review.md').write_text(text,encoding='utf-8')
    print(json.dumps({'passed':result['passed'],'output_dir':str(out),'elapsed_seconds':result['elapsed_seconds'],
                      'error':result.get('error')},ensure_ascii=False),flush=True)
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':
    main()
