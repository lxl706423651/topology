import argparse
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader, TensorDataset, Sampler
from tqdm import tqdm
import gc
import torch.nn.functional as F
class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=3, output_size=1, dropout=0.2):
        super(BiLSTMModel, self).__init__()
        
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            bidirectional=True,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        
        self.fc = nn.Linear(hidden_size * 4, output_size)  # 输出维度变为 hidden_size * 4
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # lstm_out形状: (batch_size, sequence_length, hidden_size * 2)

        # 对所有时间步的输出进行最大值和平均值操作
        max_pool = torch.max(lstm_out, dim=1)[0]  # (batch_size, hidden_size * 2)
        mean_pool = torch.mean(lstm_out, dim=1)  # (batch_size, hidden_size * 2)

        # 拼接最大值和平均值，得到 (batch_size, hidden_size * 4)
        lstm_out = torch.cat((max_pool, mean_pool), dim=1)

        lstm_out = self.dropout(lstm_out)  # Dropout
        output = self.fc(lstm_out)  # 通过全连接层
        
        return output

class MyCollateFn4:
    def __init__(self, x, y, group, symbol,symbol_dict,window_size=24,is_train=True):
        self.x = x  # 特征数据
        self.y = y  # 目标数据
        self.group = group  # 分组信息
        self.symbol = symbol  # symbol 信息
        self.symbol_matching_indices = symbol_dict
        self.window_size = window_size  # 时间步窗口大小，默认24
        self.is_train=is_train
    def huigun(self, batch_indices):
        symbol_codes = self.symbol[batch_indices]  # 获取当前批次样本的所有 symbol
        previous_indices = []
        for idx, symbol_code in zip(batch_indices, symbol_codes):
            matching_indices=self.symbol_matching_indices[symbol_code]
            previous_24_indices = matching_indices[matching_indices <= idx][-self.window_size:]
            if isinstance(previous_24_indices, np.ndarray) and len(previous_24_indices) == self.window_size:  # 确保窗口大小为 24
                previous_indices.append(previous_24_indices)
                
        return previous_indices

    def __call__(self, batch):
        batch_size = len(batch)
        
        # 获取当前批次样本的前 window_size 个时间步的索引
        previous_indices_list = self.huigun(batch)
        # 如果没有满足条件的样本，返回默认张量
        if len(previous_indices_list) == 0:
            if(self.is_train):return None,None,None
            else:return None,None,None,None
        # 合并所有满足条件的前 24 个时间步的索引
        previous_indices_tensor = torch.tensor(np.concatenate(previous_indices_list), dtype=torch.int64)

        # 批量索引并确保 x1 的维度是 (batch_size, window_size, feature_dim)
        x1 = self.x[previous_indices_tensor].to(torch.float32).view(len(previous_indices_list), self.window_size, self.x.shape[1])
        # y1 和 group1 需要从 previous_indices_list 中每个数组的最后一个索引提取
        idxs=[indices[-1] for indices in previous_indices_list]
        y1 = torch.tensor([self.y[indices[-1]] for indices in previous_indices_list], dtype=torch.float32)
        group1 = torch.tensor([self.group[indices[-1]] for indices in previous_indices_list], dtype=torch.int16)
        if(self.is_train):return x1, y1, group1
        else:return x1, y1, group1,idxs

def gz_symbol_matching_indices(symbol):
    t=np.unique(symbol)
    symbol_dict={}
    for symbol_code in t:
        symbol_dict[symbol_code]=np.where(symbol == symbol_code)[0]
    return symbol_dict

def load_and_preprocess_data(filepath):
    """
    读取 CSV 数据并进行预处理
    """
    data = pd.read_csv(filepath)
    print(data.shape)
    data = data.fillna(0)
    data['candle_begin_time'] = pd.to_datetime(data['candle_begin_time'])
    TARGET = 'Return_24'
    # 特征选择：排除不需要的列
    exclude_cols = ['candle_begin_time', 'symbol', 'is_spot', 'close', 'Return_24', 'count']
    USE_COLS = data.columns.difference(exclude_cols)
    return data, USE_COLS, TARGET

def spearman(model,dataloader_test,test_data,device="cuda"):
    model.eval()
    with torch.no_grad():
        y_pred = []
        index=[]
        y_test=[]
        for batch_X, batch_y, batch_groups,idxs in dataloader_test:
            if(batch_X==None):continue
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            outputs = model(batch_X)
            y_pred.append(outputs.cpu().numpy())
            y_test.append(batch_y.cpu().numpy())
            index.extend(idxs)

        y_pred = np.concatenate(y_pred, axis=0)
        y_test = np.concatenate(y_test, axis=0)
        y_pred = y_pred.squeeze(-1)  # 需要去掉多余的维度
        # loss = mean_squared_error(y_test, y_pred)
        # print(f"Test primary ranking metric: {loss:.4f}")

        # 计算每个时间段内 Spearman 相关性（假设 test_data 中存在 candle_begin_time 列）
        test_data_copy = test_data.iloc[index,:].copy()
        test_data_copy['preds'] = y_pred
        # 注意：这里需要 groupby 依据 'candle_begin_time' 计算相关性
        hourly_corr = test_data_copy.groupby('candle_begin_time')[['preds', 'Return_24']].corr(method='spearman').unstack().iloc[:, 1]
        mean_corr = hourly_corr.mean()
    return mean_corr
def train_and_evaluate(train_data, test_data, USE_COLS, TARGET, lr, batch_size, epochs, patience, max_epochs, device, num_workers, margin=0.33):
    """
    训练并评估模型。
    这里假设 DataFrame 中的 'candle_begin_time' 与 'symbol' 构成主键，
    损失函数只比较相同 candle_begin_time 下的排序损失，
    同时根据 candle_begin_time 列计算组别（转换为类别编码）。
    """
    start_total = time.time()

    # 1. 准备训练与测试数据
    X_train = train_data[USE_COLS].values
    y_train = train_data[TARGET].values.reshape(-1)  # 1D array
    # 将 candle_begin_time 列转换为类别编码（整数）用于分组
    group_train = train_data['candle_begin_time'].astype('category').cat.codes.values
    symbol_train = train_data['symbol'].astype('category').cat.codes.values
    # 转换为 PyTorch 张量
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    group_train_tensor = torch.tensor(group_train, dtype=torch.long)
    symbol_train_tensor = torch.tensor(symbol_train, dtype=torch.int16)
    symbol_dict=gz_symbol_matching_indices(symbol_train)
    
    X_test = test_data[USE_COLS].values
    y_test = test_data[TARGET].values.reshape(-1)
    symbol_test = test_data['symbol'].astype('category').cat.codes.values
    group_test = test_data['candle_begin_time'].astype('category').cat.codes.values
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    group_test_tensor = torch.tensor(group_test, dtype=torch.long)
    symbol_test_tensor = torch.tensor(symbol_test, dtype=torch.int16)
    symbol_test_dict=gz_symbol_matching_indices(symbol_test)
    
    # 2. 构建 DataLoader（训练时同时返回组信息）
    print('X_train_tensor:',len(X_train_tensor))
    t=time.time()
    collate_fn_train = MyCollateFn4(X_train_tensor, y_train_tensor, group_train_tensor,symbol_train,symbol_dict,window_size=24)
    dataloader_train = DataLoader(
        range(len(X_train)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=32,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        collate_fn=collate_fn_train
    )
    print('len(dataloader_train)',len(dataloader_train),'cost time:',time.time()-t)
    collate_fn_test = MyCollateFn4(X_test_tensor, y_test_tensor, group_test_tensor,symbol_test,symbol_test_dict,window_size=24,is_train=False)
    dataloader_test = DataLoader(
        range(len(X_test)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=32,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        collate_fn=collate_fn_test
    )
    # 3. 初始化模型和优化器
    input_size = X_train.shape[1]
    model = BiLSTMModel(input_size=input_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    min_loss = float('inf')
    no_improve_count = 0
    best_model_state = None
    best_spearman_corr=-1
    #print('margin',margin)
    # 4. 训练循环
    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0
        start_epoch = time.time()

        for batch_X, batch_y, batch_groups in dataloader_train:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            batch_groups = batch_groups.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze(-1)
            loss = criterion(outputs, batch_y)
            if torch.isnan(loss):
                print("Loss is NaN! Stopping training.")
                return None, None, None
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(dataloader_train)
        elapsed_epoch = time.time() - start_epoch
        
        spearman_corr =spearman(model,dataloader_test,test_data,device)
        # Early Stopping 逻辑
        if spearman_corr > best_spearman_corr:
            best_spearman_corr = spearman_corr
            no_improve_count = 0
            best_model_state = model.state_dict()  # 保存最优模型
        else:
            no_improve_count += 1

        if no_improve_count >= patience and epoch >= epochs:
            print(f"Early stopping at epoch {epoch+1-patience} with Spearman correlation {best_spearman_corr:.4f}")
            break

        if epoch % 10 == 0:
            print(f"Epoch [{epoch}/{max_epochs}], Loss: {epoch_loss:.4f}, Spearman Corr: {spearman_corr:.4f}, Time: {elapsed_epoch:.2f}s")
    if(no_improve_count< patience):
        print(f"Training at epoch {epoch+1} with Spearman correlation {spearman_corr:.4f}")
    # 5. 评估阶段：加载最佳模型并计算 MSE 与 Spearman 相关性
    model.load_state_dict(best_model_state)
    
#     model.eval()
#     with torch.no_grad():
#         y_pred = []
#         index=[]
#         y_test=[]
#         for batch_X, batch_y, batch_groups,idxs in dataloader_test:
#             if(batch_X==None):continue
#             batch_X = batch_X.to(device)
#             batch_y = batch_y.to(device)
#             outputs = model(batch_X)
#             y_pred.append(outputs.cpu().numpy())
#             y_test.append(batch_y.cpu().numpy())
#             index.extend(idxs)

#         y_pred = np.concatenate(y_pred, axis=0)
#         y_test = np.concatenate(y_test, axis=0)
#         y_pred = y_pred.squeeze(-1)  # 需要去掉多余的维度
#         loss = mean_squared_error(y_test, y_pred)
#         print(f"loss: {loss:.4f}")

#         # 计算每个时间段内 Spearman 相关性（假设 test_data 中存在 candle_begin_time 列）
#         test_data_copy = test_data.iloc[index,:].copy()
#         test_data_copy['preds'] = y_pred
#         # 注意：这里需要 groupby 依据 'candle_begin_time' 计算相关性
#         hourly_corr = test_data_copy.groupby('candle_begin_time')[['preds', TARGET]]\
#                                      .corr(method='spearman').unstack().iloc[:, 1]
#         mean_corr = hourly_corr.mean()
#         print(f"Mean Spearman correlation: {mean_corr:.4f}")
    mean_corr =spearman(model,dataloader_test,test_data,device)
    loss = mean_corr
    print(f"Mean Spearman correlation: {mean_corr:.4f}")
    
    total_time = time.time() - start_total
    print(f"Training and evaluation took {total_time:.2f} seconds")
    return model, loss, mean_corr

def main(args):
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"GPU Memory Cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")

    # 数据加载和预处理
    data, USE_COLS, TARGET = load_and_preprocess_data(args.data_path)
    TIME='candle_begin_time'
    print(f"Number of features used: {len(USE_COLS)}")
    # 基于时间分割数据集（例如：以年份为分界）
    # 2021年的作为验证集
    
    test = data[pd.to_datetime(data[TIME]) < pd.to_datetime('2022-01-01')].reset_index(drop=True)  # 测试集
    train = data[pd.to_datetime(data[TIME]) >= pd.to_datetime('2022-01-01')].reset_index(drop=True)  # 训练集
    model, rmse, mean_corr = train_and_evaluate(train, test, USE_COLS, TARGET,
                                                    lr=args.learning_rate,
                                                    batch_size=args.batch_size,
                                                    epochs=args.epochs,
                                                    patience=args.patience,
                                                    max_epochs=args.max_epochs,
                                                    device=device,
                                                    num_workers=args.num_workers)
    del test,train
    gc.collect()
    # 2022年的作为验证集
    # 转换 TIME 列为 datetime 类型（如果数据中的 TIME 列还不是 datetime 类型的话）
    test = data[(data[TIME] >= pd.to_datetime('2022-01-01')) & (data[TIME] < pd.to_datetime('2023-01-01'))].reset_index(drop=True)
    train = data[~((data[TIME] >= pd.to_datetime('2022-01-01')) & (data[TIME] < pd.to_datetime('2023-01-01')))].reset_index(drop=True)
    model, rmse, mean_corr = train_and_evaluate(train, test, USE_COLS, TARGET,
                                                    lr=args.learning_rate,
                                                    batch_size=args.batch_size,
                                                    epochs=args.epochs,
                                                    patience=args.patience,
                                                    max_epochs=args.max_epochs,
                                                    device=device,
                                                    num_workers=args.num_workers)
    del test,train
    gc.collect()

    # 2023年的作为验证集
    test = data[(data[TIME] >= pd.to_datetime('2023-01-01')) & (data[TIME] < pd.to_datetime('2024-01-01'))].reset_index(drop=True)
    train = data[~((data[TIME] >= pd.to_datetime('2023-01-01')) & (data[TIME] < pd.to_datetime('2024-01-01')))].reset_index(drop=True)
    model, rmse, mean_corr = train_and_evaluate(train, test, USE_COLS, TARGET,
                                                    lr=args.learning_rate,
                                                    batch_size=args.batch_size,
                                                    epochs=args.epochs,
                                                    patience=args.patience,
                                                    max_epochs=args.max_epochs,
                                                    device=device,
                                                    num_workers=args.num_workers)
    del test,train
    gc.collect()

    # 2024年及之后的作为验证集
    test = data[pd.to_datetime(data[TIME]) >= pd.to_datetime('2024-01-01')].reset_index(drop=True)  # 测试集
    train = data[pd.to_datetime(data[TIME]) < pd.to_datetime('2024-01-01')].reset_index(drop=True)  # 训练集
    model, rmse, mean_corr = train_and_evaluate(train, test, USE_COLS, TARGET,
                                                    lr=args.learning_rate,
                                                    batch_size=args.batch_size,
                                                    epochs=args.epochs,
                                                    patience=args.patience,
                                                    max_epochs=args.max_epochs,
                                                    device=device,
                                                    num_workers=args.num_workers)
    del test,train
    gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep Learning Project for MLP Regression")
    parser.add_argument("--data_path", type=str, default="processed_data4.csv", help="Path to the CSV data file")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer")
    parser.add_argument("--batch_size", type=int, default=6000, help="Batch size for DataLoader")
    parser.add_argument("--epochs", type=int, default=0, help="Number of epochs before early stopping can kick in")
    parser.add_argument("--patience", type=int, default=20, help="Patience for early stopping")
    parser.add_argument("--max_epochs", type=int, default=500, help="Maximum number of training epochs")
    parser.add_argument("--num_workers", type=int, default=32, help="Number of workers for DataLoader")
    parser.add_argument("--save_dir", type=str, default="./models", help="Directory to save the trained models")
    args = parser.parse_args()
    # os.makedirs(args.save_dir, exist_ok=True)
    main(args)
