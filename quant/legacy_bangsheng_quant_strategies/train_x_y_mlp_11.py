import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import time
# 读取数据
fea_all = pd.read_csv("processed_data.csv")
data=fea_all.fillna(0)
# 选取特征candle_begin_time,symbol,is_spot,close
TIME='candle_begin_time'
TIME='candle_begin_time'
TARGET='Return_24'
USE_COLS = data.columns.difference(['candle_begin_time','symbol','is_spot','close','Return_24','count'])
print(f"USE_COLS:{len(USE_COLS)}")
# 数据预处理
data['count'] = data.groupby('candle_begin_time')['symbol'].transform('count') - 1
data[TARGET] = (data.groupby('candle_begin_time')[TARGET].rank(method='min', ascending=True)-1) / data['count']
for fea in tqdm(list(USE_COLS)):
    data[fea] = (data.groupby('candle_begin_time')[fea].rank(method='min', ascending=True)-1) / data['count']

#定义mlp模型
def cs(test,train):
    t=time.time()
    X_train = train[USE_COLS].values
    y_train = train[TARGET].values.reshape(-1, 1)
    X_test = test[USE_COLS].values
    y_test = test[TARGET].values.reshape(-1, 1)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    # 4. 定义 MLP 模型
    class MLPModel(nn.Module):
        def __init__(self, input_size):
            super(MLPModel, self).__init__()
            self.fc1 = nn.Linear(input_size, 64)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, 1)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    # 5. 初始化模型
    input_size = X_train.shape[1]
    model = MLPModel(input_size)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 6. 训练模型
    epochs = 100
    min_loss = float('inf')
    patience = 20  # 允许损失不下降的最大连续轮数
    no_improve_count = 0
    max_epochs = 500  # 最大训练轮数

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor).squeeze(-1)
        loss = criterion(outputs, y_train_tensor)

        # 检查 loss 是否变成 NaN
        if torch.isnan(loss):
            print("Loss is NaN! Stopping training.")
            break

        loss.backward()
        optimizer.step()

        if loss.item() < min_loss:
            min_loss = loss.item()
            no_improve_count = 0  # 复位耐心计数器
        else:
            no_improve_count += 1

        if no_improve_count >= patience and epoch > epochs:
            print(f"Stopping early at epoch {epoch+1} due to no improvement in loss.")
            break

        if (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch+1}/{max_epochs}], Loss: {loss.item():.4f}')
    
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor).numpy().flatten().tolist()
        y_test_numpy = y_test_tensor.numpy()

        # 计算 RMSE
        rmse = np.sqrt(mean_squared_error(y_test_numpy, y_pred))
        print(f'Test RMSE: {rmse:.4f}')
        test['preds'] = y_pred
    hourly_columns= test.groupby('candle_begin_time')[['preds', 'Return_24']].corr(method='spearman').unstack().iloc[:, 1]
    print('preds', 'Return_24',hourly_columns.mean())
    print(f"cost time:{time.time()-t}")
    return hourly_columns.mean()
# 2021年的作为验证集
test = data[pd.to_datetime(data[TIME]) < pd.to_datetime('2022-01-01')].reset_index(drop=True)  # 测试集
train = data[pd.to_datetime(data[TIME]) >= pd.to_datetime('2022-01-01')].reset_index(drop=True)  # 训练集
cs(test,train)

# 2022年的作为验证集
# 转换 TIME 列为 datetime 类型（如果数据中的 TIME 列还不是 datetime 类型的话）
data[TIME] = pd.to_datetime(data[TIME])
test = data[(data[TIME] >= pd.to_datetime('2022-01-01')) & (data[TIME] < pd.to_datetime('2023-01-01'))].reset_index(drop=True)
train = data[~((data[TIME] >= pd.to_datetime('2022-01-01')) & (data[TIME] < pd.to_datetime('2023-01-01')))].reset_index(drop=True)
cs(test,train)

# 2023年的作为验证集
data[TIME] = pd.to_datetime(data[TIME])
test = data[(data[TIME] >= pd.to_datetime('2023-01-01')) & (data[TIME] < pd.to_datetime('2024-01-01'))].reset_index(drop=True)
train = data[~((data[TIME] >= pd.to_datetime('2023-01-01')) & (data[TIME] < pd.to_datetime('2024-01-01')))].reset_index(drop=True)
cs(test,train)

# 2024年及之后的作为验证集
test = data[pd.to_datetime(data[TIME]) >= pd.to_datetime('2024-01-01')].reset_index(drop=True)  # 测试集
train = data[pd.to_datetime(data[TIME]) < pd.to_datetime('2024-01-01')].reset_index(drop=True)  # 训练集
cs(test,train)