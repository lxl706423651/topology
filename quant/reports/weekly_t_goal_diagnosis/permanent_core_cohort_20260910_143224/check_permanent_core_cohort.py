"""只读核8份已审计OOF名单及安全字段连接；不拟合、不回测、不读取收益表。"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

root = Path('reports/weekly_t_cross_validation/run_20260908_235818_404243')
cache = Path('reports/weekly_t_cross_validation/cache/weekly_t_9b6b9bfb3309dd53')
out = Path(Path('/tmp/permanent_core_cohort_memo_path.txt').read_text())

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()

audit_path = root/'independent_residual_audit.json'
audit = json.loads(audit_path.read_text())
assert all(audit[k] is True for k in ('passed', 'complete', 'full_configured_scope_complete'))
identity = json.loads(Path('reports/weekly_t_cross_validation/run_20260909_182945_388303/input_sources.json').read_text())
weekly_path = cache/'weekly.pkl'
assert sha(weekly_path) == identity[str(weekly_path.resolve())]['sha256']
# 仅取来源/日历/当前已知价格；未来收益/执行端点列不参加任何计算。
columns = ['trade_week','symbol','feature_week','signal_datetime','entry_date','exit_date',
           'f_last_close','observed_last_datetime','feature_week_complete']
weekly = pd.read_pickle(weekly_path)[columns].copy()
observed = weekly[['symbol','feature_week']].drop_duplicates().sort_values(['symbol','feature_week'])
grouped = observed.groupby('symbol',sort=False).feature_week
observed['feature_source_start'] = grouped.shift(4).fillna(grouped.transform('first'))
weekly = weekly.merge(observed,on=['symbol','feature_week'],how='left',validate='one_to_one')
assert not weekly.duplicated(['trade_week','symbol']).any()
rows, proofs = [], {}
for test_year in (2023,2024,2025,2026):
    fold = json.loads((root/f'year_{test_year}/split.json').read_text())
    folder = root/f'year_{test_year}/oof_weekly_cohort'
    metadata = json.loads((folder/'cohort_metadata.json').read_text())
    assert metadata['core_years'] == fold['core_years']
    for year in fold['core_years']:
        directory = folder/f'selector_{year}'
        verified = audit['selectors'][str(directory.relative_to(root))]
        actual = json.loads((directory/'model_metadata.json').read_text())
        expected_train = sorted(set(fold['core_years'])-{year})
        expected_excluded = sorted({test_year,fold['inner_year'],year})
        assert actual['train_years'] == expected_train and actual['excluded_years'] == expected_excluded
        assert actual['training_sample_sha256'] == verified['sample_sha256']
        assert actual['feature_source_sha256'] == verified['source_sha256']
        files = {'model_bundle.joblib':verified['model_bundle_sha256'],
                 'oof_weekly_selections.csv':verified['selected_csv_sha256']}
        files.update({name+'_model.txt':value['native_sha256'] for name,value in verified['native_checks'].items()})
        assert all(sha(directory/name)==digest for name,digest in files.items())
        selected = pd.read_csv(directory/'oof_weekly_selections.csv',float_precision='round_trip')
        for col in ['trade_week','feature_week','signal_datetime']:
            selected[col] = pd.to_datetime(selected[col])
        assert selected.source_model_training_years.eq(json.dumps(expected_train,separators=(',',':'))).all()
        assert selected.source_model_sha256.eq(files['model_bundle.joblib']).all()
        joined = selected.merge(weekly,on=['trade_week','symbol'],how='left',validate='one_to_one',suffixes=('','_cache'),indicator=True)
        assert joined._merge.eq('both').all()
        assert joined.feature_week.eq(joined.feature_week_cache).all()
        assert joined.signal_datetime.eq(joined.signal_datetime_cache).all()
        assert joined.signal_datetime.lt(joined.entry_date+pd.Timedelta(hours=9,minutes=30)).all()
        assert joined.observed_last_datetime.le(joined.signal_datetime).all()
        assert (np.isfinite(joined.f_last_close)&joined.f_last_close.gt(0)).all()
        # 严格训练用的入场来源筛选：先排除完全不属于core的原始来源区间。
        # 这里只量化名单，不据此实际规划/拟合；epoch cash_state依赖仍需新账本记录。
        valid = joined.feature_source_start.dt.year.isin(fold['core_years'])
        for excluded in (test_year,fold['inner_year']):
            start,end = pd.Timestamp(excluded,1,1),pd.Timestamp(excluded+1,1,1)
            valid &= ~((joined.feature_source_start<end)&(joined.signal_datetime>=start))
        # 仅对原预测文件的键做供体来源机械检查，不读取pred_return或任何label。
        prediction_path = directory/'oof_weekly_predictions.csv'
        donors = pd.read_csv(prediction_path,usecols=['trade_week','symbol'])
        donors.trade_week = pd.to_datetime(donors.trade_week)
        donors = donors.merge(weekly[['trade_week','symbol','feature_source_start']],
                              on=['trade_week','symbol'],how='left',validate='one_to_one')
        assert donors.feature_source_start.notna().all()
        ranked_start = joined.trade_week.map(donors.groupby('trade_week').feature_source_start.min())
        strict = valid.copy()
        rank_strict = ranked_start.dt.year.isin(fold['core_years'])
        for excluded in (test_year,fold['inner_year']):
            start = pd.Timestamp(excluded,1,1)-pd.Timedelta(days=7)
            end = pd.Timestamp(excluded+1,1,1)+pd.Timedelta(days=7)
            entry_clock = joined.entry_date+pd.Timedelta(hours=9,minutes=30)
            strict &= ~((joined.feature_source_start<end)&(entry_clock>=start))
            rank_strict &= ~((ranked_start<end)&(entry_clock>=start))
        rows.append({'outer_year':test_year,'inner_year':fold['inner_year'],'cohort_target_year':year,
                     'selector_train_year':expected_train[0],'selected_stockweeks':len(selected),
                     'selected_weeks':int(selected.trade_week.nunique()),'missing_join_rows':int(joined._merge.ne('both').sum()),
                     'source_outside_core_stockweeks':int((~valid).sum()),
                     'source_outside_core_weeks':int(joined.loc[~valid,'trade_week'].nunique()),
                     'first_selection_signal':str(joined.signal_datetime.min()),
                     'first_selection_source_start':str(joined.feature_source_start.min()),
                     'first_source_eligible_entry_date':str(joined.loc[valid,'entry_date'].min()),
                     'own_source_or_7d_embargo_excluded_stockweeks':int((~strict).sum()),
                     'first_own_source_7d_clean_entry_date':str(joined.loc[strict,'entry_date'].min()),
                     'rank_donor_or_7d_embargo_excluded_stockweeks':int((~rank_strict).sum()),
                     'first_rank_source_7d_clean_entry_date':str(joined.loc[rank_strict,'entry_date'].min()),
                     'old_middle_day_cohort_rows':next(item['cohort_rows_before_t_label_purge'] for item in metadata['selectors'] if item['target_year']==year),
                     'entry_date_missing_in_saved_selection':'entry_date' not in selected,
                     'previous_close_missing_in_saved_selection':'universe_previous_close' not in selected})
        proofs[str(directory)]={'all_declared_selector_identities_match':True,'train_years':actual['train_years'],
                                'excluded_years':actual['excluded_years'],'training_rows':actual['training_rows'],
                                'training_sample_sha256':actual['training_sample_sha256'],
                                'feature_source_sha256':actual['feature_source_sha256'],
                                'files_sha256':files,'model_metadata_sha256':sha(directory/'model_metadata.json'),
                                'prediction_keys_observed_sha256':sha(prediction_path),
                                'rank_donor_keys_checks_only_not_prediction_value_reaudit':True,
                                'no_fit_or_prediction_performed':True}

pd.DataFrame(rows).to_csv(out/'selector_join_summary.csv',index=False)
result={'passed':True,'classification':'read_only_training_cohort_feasibility_no_account_run_no_fit_no_return_selection',
        'selector_count':len(rows),'selected_stockweeks':sum(row['selected_stockweeks'] for row in rows),
        'all_selected_joined_exactly':True,'future_returns_used':False,'v13_results_read':False,
        'source_interval_method':'本股前4实际观测周的最早周一，与已审计provenance定义一致；未重做整个universe/模型审计',
        'does_not_claim_fresh_full_model_audit':True,'summary':rows,'selector_proofs':proofs,
        'inputs_sha256':{str(path):sha(path) for path in (audit_path,weekly_path,cache/'metadata.json')},
        'source_sha256':{str(path):sha(path) for path in [Path('quant_backtest/weekly_t_timing_data.py'),
                             Path('quant_backtest/weekly_t_cohort.py'),Path('quant_backtest/weekly_t_provenance.py'),
                             Path('quant_backtest/dynamic_t_engine.py'),Path('quant_backtest/balanced_t_engine.py')]}}
(out/'feasibility_check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(pd.DataFrame(rows).to_string(index=False))
