import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import copy

from sklearn.linear_model import LogisticRegression,ElasticNet
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# 评价方法
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split,StratifiedKFold

import lightgbm as lgb
#from catboost import CatBoostClassifier
from tqdm import tqdm
import time 
import datetime

#超参数设置
k=10
# 读取数据
fea_df = pd.read_pickle('fea_df2.pkl')
data=fea_df


# 选取特征candle_begin_time,symbol,is_spot,close
TIME='candle_begin_time'
TARGET='Return_24'
USE_COLS = data.columns.difference(['candle_begin_time','symbol','is_spot','close','Return_24'])
# 数据预处理

data['count'] = data.groupby('candle_begin_time')['symbol'].transform('count') - 1
data[TARGET] = (data.groupby('candle_begin_time')[TARGET].rank(method='min', ascending=True)-1) / data['count']
for fea in tqdm(list(USE_COLS)):
    # 对每个特征按 'candle_begin_time' 分组，计算排名和归一化
    data[fea] = (data.groupby('candle_begin_time')[fea].rank(method='min', ascending=True)-1) / data['count']

#增加特征
data['weekday'] = data['candle_begin_time'].dt.weekday
data['hour'] = data['candle_begin_time'].dt.hour
USE_COLS = data.columns.difference(['candle_begin_time','symbol','is_spot','close','Return_24','count'])
for fea in USE_COLS:
    if data[fea].dtypes == 'object':
        data[fea] = data[fea].astype('float')
        print(f'{fea}已转换!')

# 2021年的作为验证集
test = data[pd.to_datetime(data[TIME]) < pd.to_datetime('2022-01-01')].reset_index(drop=True)  # 测试集
train = data[pd.to_datetime(data[TIME]) >= pd.to_datetime('2022-01-01')].reset_index(drop=True)  # 训练集
X_train = train[USE_COLS]
y_train = train[TARGET]
X_val = test[USE_COLS]
y_val = test[TARGET]
t=time.time()
lgb_train = lgb.Dataset(X_train, label = y_train)
lgb_eval = lgb.Dataset(X_val, label = y_val)
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',  # 将目标设置为回归
    'metric': 'rmse',  # 使用L2误差作为评估指标，或者你可以选择 'mae' 或 'rmse'
    'num_leaves': 128,
    'max_depth': 20,
    'min_data_in_leaf': 50,
    'learning_rate': 0.01,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'lambda_l1': 1,
    'lambda_l2': 1,  # 正则化，越小正则化效果越强
    'min_gain_to_split': 0.2,
    'verbose': -1,  # 关闭日志输出
    # 'device': 'gpu',  # 如果需要使用 GPU 训练，取消注释
}

# 训练回归模型
lgb_clf = lgb.train(
    params,
    lgb_train,
    num_boost_round=10000,
    valid_sets=lgb_eval,
    callbacks=[
        lgb.early_stopping(stopping_rounds=200),  # 提前停止
        lgb.log_evaluation(period=50)  # 每 50 次迭代打印一次日志
    ]
)
print(time.time()-t)
preds = lgb_clf.predict(X_val)
test['preds']=preds
hourly_columns= test.groupby('candle_begin_time')[['preds', 'Return_24']].corr(method='spearman').unstack().iloc[:, 1]
print('preds', 'Return_24',hourly_columns.mean())

# 2022年的作为验证集
# 转换 TIME 列为 datetime 类型（如果数据中的 TIME 列还不是 datetime 类型的话）
data[TIME] = pd.to_datetime(data[TIME])
# 选择2022年整年的数据作为测试集
test = data[(data[TIME] >= pd.to_datetime('2022-01-01')) & (data[TIME] < pd.to_datetime('2023-01-01'))].reset_index(drop=True)
# 其余的数据作为训练集
train = data[~((data[TIME] >= pd.to_datetime('2022-01-01')) & (data[TIME] < pd.to_datetime('2023-01-01')))].reset_index(drop=True)
X_train = train[USE_COLS]
y_train = train[TARGET]
X_val = test[USE_COLS]
y_val = test[TARGET]
t=time.time()
lgb_train = lgb.Dataset(X_train, label = y_train)
lgb_eval = lgb.Dataset(X_val, label = y_val)
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',  # 将目标设置为回归
    'metric': 'rmse',  # 使用L2误差作为评估指标，或者你可以选择 'mae' 或 'rmse'
    'num_leaves': 128,
    'max_depth': 20,
    'min_data_in_leaf': 50,
    'learning_rate': 0.01,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'lambda_l1': 1,
    'lambda_l2': 1,  # 正则化，越小正则化效果越强
    'min_gain_to_split': 0.2,
    'verbose': -1,  # 关闭日志输出
    # 'device': 'gpu',  # 如果需要使用 GPU 训练，取消注释
}

# 训练回归模型
lgb_clf = lgb.train(
    params,
    lgb_train,
    num_boost_round=10000,
    valid_sets=lgb_eval,
    callbacks=[
        lgb.early_stopping(stopping_rounds=200),  # 提前停止
        lgb.log_evaluation(period=50)  # 每 50 次迭代打印一次日志
    ]
)
print(time.time()-t)
preds = lgb_clf.predict(X_val)
test['preds']=preds
hourly_columns= test.groupby('candle_begin_time')[['preds', 'Return_24']].corr(method='spearman').unstack().iloc[:, 1]
print('preds', 'Return_24',hourly_columns.mean())

# 2023年的作为验证集
data[TIME] = pd.to_datetime(data[TIME])
test = data[(data[TIME] >= pd.to_datetime('2023-01-01')) & (data[TIME] < pd.to_datetime('2024-01-01'))].reset_index(drop=True)
train = data[~((data[TIME] >= pd.to_datetime('2023-01-01')) & (data[TIME] < pd.to_datetime('2024-01-01')))].reset_index(drop=True)
X_train = train[USE_COLS]
y_train = train[TARGET]
X_val = test[USE_COLS]
y_val = test[TARGET]
t=time.time()
lgb_train = lgb.Dataset(X_train, label = y_train)
lgb_eval = lgb.Dataset(X_val, label = y_val)
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',  # 将目标设置为回归
    'metric': 'rmse',  # 使用L2误差作为评估指标，或者你可以选择 'mae' 或 'rmse'
    'num_leaves': 128,
    'max_depth': 20,
    'min_data_in_leaf': 50,
    'learning_rate': 0.01,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'lambda_l1': 1,
    'lambda_l2': 1,  # 正则化，越小正则化效果越强
    'min_gain_to_split': 0.2,
    'verbose': -1,  # 关闭日志输出
    # 'device': 'gpu',  # 如果需要使用 GPU 训练，取消注释
}

# 训练回归模型
lgb_clf = lgb.train(
    params,
    lgb_train,
    num_boost_round=10000,
    valid_sets=lgb_eval,
    callbacks=[
        lgb.early_stopping(stopping_rounds=200),  # 提前停止
        lgb.log_evaluation(period=50)  # 每 50 次迭代打印一次日志
    ]
)
print(time.time()-t)
preds = lgb_clf.predict(X_val)
test['preds']=preds
hourly_columns= test.groupby('candle_begin_time')[['preds', 'Return_24']].corr(method='spearman').unstack().iloc[:, 1]
print('preds', 'Return_24',hourly_columns.mean())

# 2024年及之后的作为验证集
test = data[pd.to_datetime(data[TIME]) >= pd.to_datetime('2024-01-01')].reset_index(drop=True)  # 测试集
train = data[pd.to_datetime(data[TIME]) < pd.to_datetime('2024-01-01')].reset_index(drop=True)  # 训练集
X_train = train[USE_COLS]
y_train = train[TARGET]
X_val = test[USE_COLS]
y_val = test[TARGET]
t=time.time()
lgb_train = lgb.Dataset(X_train, label = y_train)
lgb_eval = lgb.Dataset(X_val, label = y_val)
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',  # 将目标设置为回归
    'metric': 'rmse',  # 使用L2误差作为评估指标，或者你可以选择 'mae' 或 'rmse'
    'num_leaves': 128,
    'max_depth': 20,
    'min_data_in_leaf': 50,
    'learning_rate': 0.01,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'lambda_l1': 1,
    'lambda_l2': 1,  # 正则化，越小正则化效果越强
    'min_gain_to_split': 0.2,
    'verbose': -1,  # 关闭日志输出
    # 'device': 'gpu',  # 如果需要使用 GPU 训练，取消注释
}

# 训练回归模型
lgb_clf = lgb.train(
    params,
    lgb_train,
    num_boost_round=10000,
    valid_sets=lgb_eval,
    callbacks=[
        lgb.early_stopping(stopping_rounds=200),  # 提前停止
        lgb.log_evaluation(period=50)  # 每 50 次迭代打印一次日志
    ]
)
print(time.time()-t)
preds = lgb_clf.predict(X_val)
test['preds']=preds
hourly_columns= test.groupby('candle_begin_time')[['preds', 'Return_24']].corr(method='spearman').unstack().iloc[:, 1]
print('preds', 'Return_24',hourly_columns.mean())