"""人工CSV年度切片与当前分钟只读overlay的必要回归，不连接交易账户。"""
from __future__ import annotations
import hashlib
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from quant_backtest.balanced_t_manual import prepare_manual_context, ManualRecoveryStore
from quant_backtest.weekly_t_engine import build_recovery_store


def read_spawned_overlay(store, timestamp, symbol):
    return store.get_quote(pd.Timestamp(timestamp),symbol)


def order(timestamp,symbol='SH.600000'):
    return {'datetime':timestamp,'signal_datetime':'2024-12-31 15:00','symbol':symbol,'shares':100,'reason':'user_add'}


def bar(timestamp,close=10.):
    return {'datetime':pd.Timestamp(timestamp),'symbol':'SH.600000','open':close,'high':close,'low':close,
            'close':close,'volume':1000.,'amount':10000.}


class BalancedManualTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.csv=self.root/'manual.csv';self.days=pd.bdate_range('2025-01-06','2025-01-10')
        self.cfg={'manual_buys_csv':str(self.csv),'data_root':str(self.root/'raw'),'start_date':'2023-01-01',
                  'end_date':'2026-04-30','lot_size':100,'manual_supported_symbols':['SH.600000']}
        self.base=build_recovery_store(pd.DataFrame([bar('2025-01-06 14:55',11.)]))

    def tearDown(self):
        self.temp.cleanup()

    def write_orders(self,rows):
        pd.DataFrame(rows).to_csv(self.csv,index=False)

    def raw_file(self,rows):
        path=self.root/'raw/2025/SH.600000.csv';path.parent.mkdir(parents=True,exist_ok=True)
        pd.DataFrame(rows).drop(columns=['symbol']).to_csv(path,index=False)
        return path

    def test_null_fast_path_does_not_read_sources(self):
        inputs={}
        with patch('pandas.read_csv',side_effect=AssertionError('must not read')):
            cfg,entries,store,audit=prepare_manual_context({'manual_buys_csv':None},2025,self.days,self.base,inputs)
        self.assertIsNone(entries);self.assertIs(store,self.base);self.assertFalse(audit['configured']);self.assertEqual(inputs,{})

    def test_year_and_smoke_slice_keeps_invalid_in_scope_dates_visible(self):
        self.write_orders([order('2024-12-31 15:00'),order('2025-01-06 09:30'),order('2025-01-13 09:30'),order('2026-01-05 09:30')])
        # 首条外部指令本身也须具有早于执行的信号。
        frame=pd.read_csv(self.csv);frame.loc[0,'signal_datetime']='2024-12-30 15:00';frame.to_csv(self.csv,index=False)
        inputs={};cfg,entries,store,audit=prepare_manual_context(self.cfg,2025,self.days,self.base,inputs)
        self.assertIsNone(cfg['manual_buys_csv']);self.assertEqual(len(entries),1);self.assertEqual(audit['rows_outside_scope'],3)
        self.assertEqual(audit['daily_quote_requests'],1);self.assertIs(store,self.base)
        self.assertEqual(inputs[str(self.csv.resolve())]['sha256'],hashlib.sha256(self.csv.read_bytes()).hexdigest())
        for timestamp in ['2025-01-05 09:30','2025-01-07 12:05',None]:
            with self.subTest(timestamp=timestamp):
                self.write_orders([order(timestamp)])
                with self.assertRaises(ValueError):prepare_manual_context(self.cfg,2025,self.days,self.base,{})

    def test_exact_outside_cache_quote_and_missing_minute_never_use_future(self):
        self.write_orders([order('2025-01-06 10:08'),order('2025-01-06 10:09')])
        raw=self.raw_file([bar('2025-01-06 10:09',10.),bar('2025-01-06 10:10',99.)])
        before=hashlib.sha256(raw.read_bytes()).hexdigest();inputs={}
        _,entries,store,audit=prepare_manual_context(self.cfg,2025,self.days,self.base,inputs)
        self.assertEqual(len(entries),2);self.assertEqual(audit['raw_minutes_found'],1);self.assertEqual(audit['raw_minutes_missing'],1)
        self.assertIsNone(store.get_quote(pd.Timestamp('2025-01-06 10:08'),'SH.600000'))
        value=store.get_quote(pd.Timestamp('2025-01-06 10:09'),'SH.600000');self.assertEqual(value['close'],10.)
        value['close']=999.;self.assertEqual(store.get_quote(pd.Timestamp('2025-01-06 10:09'),'SH.600000')['close'],10.)
        self.assertEqual(store.get_quote(pd.Timestamp('2025-01-06 14:55'),'SH.600000')['close'],11.)
        self.assertEqual(inputs[str(raw.resolve())]['sha256'],before);self.assertEqual(hashlib.sha256(raw.read_bytes()).hexdigest(),before)

    def test_overlay_spawn_serialization_and_missing_raw_file(self):
        self.write_orders([order('2025-01-06 10:09')]);self.raw_file([bar('2025-01-06 10:09')])
        _,_,store,_=prepare_manual_context(self.cfg,2025,self.days,self.base,{})
        self.assertIsInstance(store,ManualRecoveryStore)
        with ProcessPoolExecutor(max_workers=1,mp_context=mp.get_context('spawn')) as pool:
            value=pool.submit(read_spawned_overlay,store,'2025-01-06 10:09','SH.600000').result(timeout=30)
        self.assertEqual(value['close'],10.)
        self.cfg['data_root']=str(self.root/'absent')
        _,entries,store,audit=prepare_manual_context(self.cfg,2025,self.days,self.base,{})
        self.assertEqual(len(entries),1);self.assertEqual(len(audit['missing_raw_files']),1)
        self.assertIsNone(store.get_quote(pd.Timestamp('2025-01-06 10:09'),'SH.600000'))

    def test_market_membership_lots_and_existing_zero_quote_are_not_bypassed(self):
        for symbol,shares in [('SZ.300001',100),('SH.600001',100),('SH.600000',50)]:
            row=order('2025-01-06 10:09',symbol);row['shares']=shares;self.write_orders([row])
            with self.subTest(symbol=symbol,shares=shares),self.assertRaises(ValueError):
                prepare_manual_context(self.cfg,2025,self.days,self.base,{})
        self.write_orders([order('2025-01-06 10:09')])
        zero=bar('2025-01-06 10:09');zero['volume']=0.;zero['amount']=0.
        base=build_recovery_store(pd.DataFrame([zero]));inputs={}
        _,entries,store,audit=prepare_manual_context(self.cfg,2025,self.days,base,inputs)
        self.assertIs(store,base);self.assertEqual(audit['existing_current_cache_quotes_used'],1)
        self.assertEqual(audit['requested_raw_minutes'],0);self.assertEqual(len(entries),1)
        self.assertEqual(store.get_quote(pd.Timestamp('2025-01-06 10:09'),'SH.600000')['volume'],0.)


if __name__=='__main__':
    unittest.main(verbosity=2)
