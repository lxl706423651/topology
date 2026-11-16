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
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

# 定义多层感知机模型
class MLPModel(nn.Module):
    def __init__(self, input_size):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def load_and_preprocess_data(filepath):
    """
    读取 CSV 数据并进行预处理
    """
    data = pd.read_csv(filepath)
    data = data.fillna(0)
    data['candle_begin_time'] = pd.to_datetime(data['candle_begin_time'])
    
    # 计算每个时间点的 symbol 数量（减去 1 以用于归一化排名）
    data['count'] = data.groupby('candle_begin_time')['symbol'].transform('count') - 1
    TARGET = 'Return_24'
    
    # 对目标变量按时间组内排名归一化
    data[TARGET] = (data.groupby('candle_begin_time')[TARGET].rank(method='min', ascending=True) - 1) / data['count']
    # 特征选择：排除不需要的列
    exclude_cols = ['candle_begin_time', 'symbol', 'is_spot', 'close', 'Return_24', 'count']
    USE_COLS = data.columns.difference(exclude_cols)
    # 归一化数据
    data[USE_COLS] = StandardScaler().fit_transform(data[USE_COLS])
    #data[TARGET] = StandardScaler().fit_transform(data[[TARGET]])
    
    return data, USE_COLS, TARGET

def train_and_evaluate(train_data, test_data, USE_COLS, TARGET, lr, batch_size, epochs, patience, max_epochs, device, num_workers):
    """
    训练并评估模型
    """
    start_total = time.time()
    # 准备训练和测试数据
    X_train = train_data[USE_COLS].values
    y_train = train_data[TARGET].values.reshape(-1, 1)
    X_test = test_data[USE_COLS].values
    y_test = test_data[TARGET].values.reshape(-1, 1)

    # 转换为 PyTorch 张量
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # 构建 DataLoader
    dataset_train = TensorDataset(X_train_tensor, y_train_tensor)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, 
                                  num_workers=num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    
    # 初始化模型、损失函数和优化器（使用命令行传入的学习率 lr）
    input_size = X_train.shape[1]
    model = MLPModel(input_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    min_loss = float('inf')
    no_improve_count = 0
    best_model_state = None

    # 训练过程
    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0
        start_epoch = time.time()

        for batch_X, batch_y in dataloader_train:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
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

        # 如果损失降低，保存模型状态
        if epoch_loss < min_loss:
            min_loss = epoch_loss
            no_improve_count = 0
            best_model_state = model.state_dict()
        else:
            no_improve_count += 1

        # 达到早停条件后退出训练
        if no_improve_count >= patience and epoch >= epochs:
            print(f"Early stopping at epoch {epoch+1-patience} with loss {epoch_loss:.4f}")
            break

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{max_epochs}], Loss: {epoch_loss:.4f}, Time: {elapsed_epoch:.2f}s")

    # 加载最佳模型进行评估
    model.load_state_dict(best_model_state)
    model.eval()
    with torch.no_grad():
        X_test_tensor = X_test_tensor.to(device)
        y_pred = model(X_test_tensor).cpu().numpy().flatten()
        y_test_numpy = y_test_tensor.numpy().flatten()
        mse = mean_squared_error(y_test_numpy, y_pred)
        print(f"Test MSE: {mse:.4f}")

        # 添加预测结果，并计算每个时间段内 Spearman 相关性
        test_data = test_data.copy()
        test_data['preds'] = y_pred
        hourly_corr = test_data.groupby('candle_begin_time')[['preds', TARGET]].corr(method='spearman').unstack().iloc[:, 1]
        mean_corr = hourly_corr.mean()
        print(f"Mean Spearman correlation: {mean_corr:.4f}")

    total_time = time.time() - start_total
    print(f"Training and evaluation took {total_time:.2f} seconds")
    return model, mse, mean_corr

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
    print(f"Number of features used: {len(USE_COLS)}")

    # 基于时间分割数据集（例如：以年份为分界）
    splits = [
        # 用2021年的数据作为测试集，其余数据作为训练集
        ("2021", 
         data[pd.to_datetime(data['candle_begin_time']) < pd.to_datetime('2022-01-01')].reset_index(drop=True),
         data[pd.to_datetime(data['candle_begin_time']) >= pd.to_datetime('2022-01-01')].reset_index(drop=True)),

        # 用2022年的数据作为测试集，其余数据作为训练集
        ("2022", 
         data[(data['candle_begin_time'] >= pd.to_datetime('2022-01-01')) & 
              (data['candle_begin_time'] < pd.to_datetime('2023-01-01'))].reset_index(drop=True),
         data[~((data['candle_begin_time'] >= pd.to_datetime('2022-01-01')) & 
                (data['candle_begin_time'] < pd.to_datetime('2023-01-01')))].reset_index(drop=True)),

        # 用2023年的数据作为测试集，其余数据作为训练集
        ("2023", 
         data[(data['candle_begin_time'] >= pd.to_datetime('2023-01-01')) & 
              (data['candle_begin_time'] < pd.to_datetime('2024-01-01'))].reset_index(drop=True),
         data[~((data['candle_begin_time'] >= pd.to_datetime('2023-01-01')) & 
                (data['candle_begin_time'] < pd.to_datetime('2024-01-01')))].reset_index(drop=True)),

        # 用2024年及之后的数据作为测试集，其余数据作为训练集
        ("2024+", 
         data[pd.to_datetime(data['candle_begin_time']) >= pd.to_datetime('2024-01-01')].reset_index(drop=True),
         data[pd.to_datetime(data['candle_begin_time']) < pd.to_datetime('2024-01-01')].reset_index(drop=True))
    ]

    for split_name, test_data, train_data in splits:
        print(f"\n=== Training for split: {split_name} ===")
        model, rmse, mean_corr = train_and_evaluate(train_data, test_data, USE_COLS, TARGET,
                                                    lr=args.learning_rate,
                                                    batch_size=args.batch_size,
                                                    epochs=args.epochs,
                                                    patience=args.patience,
                                                    max_epochs=args.max_epochs,
                                                    device=device,
                                                    num_workers=args.num_workers)
        # if model is not None:
        #     model_path = os.path.join(args.save_dir, f"model_{split_name}.pth")
        #     torch.save(model.state_dict(), model_path)
        #     print(f"Saved model to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep Learning Project for MLP Regression")
    parser.add_argument("--data_path", type=str, default="processed_data.csv", help="Path to the CSV data file")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate for the optimizer")
    parser.add_argument("--batch_size", type=int, default=8192, help="Batch size for DataLoader")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs before early stopping can kick in")
    parser.add_argument("--patience", type=int, default=20, help="Patience for early stopping")
    parser.add_argument("--max_epochs", type=int, default=500, help="Maximum number of training epochs")
    parser.add_argument("--num_workers", type=int, default=48, help="Number of workers for DataLoader")
    parser.add_argument("--save_dir", type=str, default="./models", help="Directory to save the trained models")
    args = parser.parse_args()

    # os.makedirs(args.save_dir, exist_ok=True)
    main(args)
