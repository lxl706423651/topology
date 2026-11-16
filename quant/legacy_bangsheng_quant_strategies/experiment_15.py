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

def calc_loss(y, y_hat, groups, margin=0.33):
    """
    计算排序 hinge loss：
    对于同一 candle_begin_time（组内）的所有 symbol，
    生成所有两两组合，计算 hinge loss：
        hinge = relu(-label * (pred_left - pred_right) + margin)
    其中 label = +1 表示 y_left > y_right，反之为 -1。
    最终对所有组内 loss 取平均。
    
    参数：
      y: 真实值，1D tensor
      y_hat: 预测值，1D tensor，与 y 形状相同
      groups: 1D tensor，记录每个样本所属的分组（candle_begin_time 对应的类别编码）
      margin: hinge loss 中的 margin 参数
    """
    device = y.device
    unique_groups = torch.unique(groups)
    group_losses = []
    for g in unique_groups:
        # 取出该组内的索引
        indices = (groups == g).nonzero(as_tuple=True)[0]
        if indices.numel() < 2:
            # 少于两个样本无法形成组合，跳过该组
            continue
        # 选择该组内对应的真实值和预测值
        y_g = y[indices]
        y_hat_g = y_hat[indices]
        n = y_g.shape[0]
        # 生成该组内所有两两组合（注意这里内部组合索引是相对于组内的）
        comb = combinations(n, device)
        # 根据组合索引获得组内样本的真实值与预测值
        y_left = y_g[comb[:, 0]]
        y_right = y_g[comb[:, 1]]
        pred_left = y_hat_g[comb[:, 0]]
        pred_right = y_hat_g[comb[:, 1]]
        # 定义 label：若 y_left > y_right，则 label=+1，否则 -1
        label = 2 * (y_left > y_right).int() - 1
        # 计算 hinge loss：如果预测顺序正确则差距超过 margin，loss 为 0
        loss = nn.functional.relu(-label * (pred_left - pred_right) + margin)
        group_losses.append(loss.mean())
    #print('calc_loss',total,time.time()-t)
    if group_losses:
        return torch.stack(group_losses).mean()
    else:
        # 如果所有组都没有足够样本，则返回 0 loss
        return torch.tensor(0.0, device=device)
    
# def calc_loss(y, y_hat, groups, margin=0.33):
#     """
#     矩阵运算实现的排序 hinge loss，与原函数等价：
#       对于每个组（由 groups 指定），生成所有 i<j 的样本对，
#       并计算 loss = relu(-label * (y_hat_i - y_hat_j) + margin)，
#       然后先计算各组内均值，再对所有组均值取平均。
#     """
#     device = y.device
#     n = y.shape[0]
    
#     # 1. 计算所有成对差分
#     diff_y = y.unsqueeze(1) - y.unsqueeze(0)         # 形状 [n, n]
#     diff_y_hat = y_hat.unsqueeze(1) - y_hat.unsqueeze(0) # 形状 [n, n]
    
#     # 2. 生成排序标签: y_i > y_j 对应 +1，否则 -1
#     label = 2 * (diff_y > 0).int() - 1                # 形状 [n, n]
    
#     # 3. 计算 hinge loss 矩阵
#     loss_matrix = F.relu(- label * diff_y_hat + margin)  # 形状 [n, n]
    
#     # 4. 构造同组掩码，仅考虑组内样本对，且只取上三角部分（i<j）
#     group_equal = (groups.unsqueeze(1) == groups.unsqueeze(0))
#     upper_mask = torch.triu(torch.ones_like(loss_matrix, dtype=torch.bool), diagonal=1)
#     mask = group_equal & upper_mask
    
#     # 5. 提取符合条件的 pair loss 以及对应的组号（取 i 或 j 均可）
#     indices_i, _ = mask.nonzero(as_tuple=True)
#     if indices_i.numel() == 0:
#         return torch.tensor(0.0, device=device)
    
#     # 注意：由于同一对的两个样本属于同一组，所以 groups[indices_i] 就是该 pair 的组号
#     groups_pairs = groups[indices_i]
#     pair_losses = loss_matrix[mask]
    
#     # 6. 利用 scatter 计算每个组内的均值
#     unique_groups, inverse_indices = torch.unique(groups_pairs, return_inverse=True)
#     group_loss_sum = torch.zeros_like(unique_groups, dtype=torch.float, device=device)
#     group_counts = torch.zeros_like(unique_groups, dtype=torch.float, device=device)
    
#     group_loss_sum = group_loss_sum.scatter_add_(0, inverse_indices, pair_losses)
#     group_counts = group_counts.scatter_add_(0, inverse_indices, torch.ones_like(pair_losses))
    
#     # 防止除以 0
#     group_loss_mean = group_loss_sum / group_counts.clamp(min=1)
    
#     # 7. 对所有组取平均
#     final_loss = group_loss_mean.mean()
#     return final_loss

class MyCollateFn:
    def __init__(self, x,y,group):
        self.x=x
        self.y=y
        self.group=group
        
    def __call__(self, batch):
        idxs = [item for item in batch]
        x1 = torch.FloatTensor(self.x[idxs])
        y1 = torch.FloatTensor(self.y[idxs])
        group1 = torch.FloatTensor(self.group[idxs])
        return x1, y1, group1
class GroupedBatchSampler(Sampler):
    """
    自定义批采样器，确保每个 batch 只包含同一个 group。
    """
    def __init__(self, group_tensor, batch_size, shuffle=True):
        """
        参数：
          group_tensor: 1D Tensor，表示每个样本的分组 ID
          batch_size: 每个 batch 的最大样本数
          shuffle: 是否在每个 epoch 打乱组内样本顺序
        """
        self.group_tensor = group_tensor.cpu().numpy()  # 转为 numpy 方便索引
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.group_indices = self._create_group_indices()

    def _create_group_indices(self):
        """
        将数据按 group 分组，返回字典 {group_id: [sample_indices]}
        """
        t=time.time()
        group_dict = {}
        for idx, group in enumerate(self.group_tensor):
            if group not in group_dict:
                group_dict[group] = []
            group_dict[group].append(idx)

        # 生成 batch 索引列表
        batch_indices = []
        batch=[]
        for group, indices in group_dict.items():
            # 分批
            if(len(batch)+len(indices)<=self.batch_size):
                batch.extend(indices)
            else:
                batch.extend(indices[:(self.batch_size-len(batch))])
                batch_indices.append(batch)
                batch=indices[(self.batch_size-len(batch)):]
        batch_indices.append(batch)
        if self.shuffle:
            np.random.shuffle(batch_indices)  # 整个 batch 序列打乱
        print(len(batch_indices),len(batch_indices)*self.batch_size)
        print('GroupedBatchSampler cost time:',time.time()-t)
        return batch_indices

    def __len__(self):
        return len(self.group_indices)

    def __iter__(self):
        """
        返回每个 batch 的索引列表
        """
        for batch in self.group_indices:
            yield batch

# 定义多层感知机模型
class MLPmodel(nn.Module):
    """ adapted from https://github.com/CUAI/CorrectAndSmooth/blob/master/gen_models.py """
    def __init__(self, d_in, n_classes=1, d_hidden=256, n_layers=3, dropout=0.2, **kwargs):
        super().__init__()
        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.act = nn.ReLU()
        if n_layers == 1:
            # just linear layer i.e. logistic regression
            self.lins.append(nn.Linear(d_in, n_classes))
        else:
            self.lins.append(nn.Linear(d_in, d_hidden))
            self.bns.append(nn.BatchNorm1d(d_hidden))
            for _ in range(n_layers - 2):
                self.lins.append(nn.Linear(d_hidden, d_hidden))
                self.bns.append(nn.BatchNorm1d(d_hidden))
            self.lins.append(nn.Linear(d_hidden, n_classes))
        self.dropout = dropout

    def forward(self, x):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = self.bns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x

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

def combinations(N, device):
    sample_size=int(N/2)
    ind = torch.arange(N, device=device)
    comb = torch.combinations(ind, r=2)
    indices = torch.randperm(comb.shape[0], device=device)[:sample_size]
    comb_sampled = comb[indices]
    return comb_sampled

#####################################################
# 修改后的排序 hinge loss：按 candle_begin_time 分组
#####################################################



#####################################################
# 训练与评估函数（使用按组计算的排序损失）
#####################################################
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
    X_test = test_data[USE_COLS].values
    y_test = test_data[TARGET].values.reshape(-1)

    # 将 candle_begin_time 列转换为类别编码（整数）用于分组
    group_train = train_data['candle_begin_time'].astype('category').cat.codes.values
    group_test = test_data['candle_begin_time'].astype('category').cat.codes.values

    # 转换为 PyTorch 张量
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    group_train_tensor = torch.tensor(group_train, dtype=torch.long)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    group_test_tensor = torch.tensor(group_test, dtype=torch.long)

    # 2. 构建 DataLoader（训练时同时返回组信息）
    print('X_train_tensor:',len(X_train_tensor))
    t=time.time()
    batch_sampler = GroupedBatchSampler(group_train_tensor, batch_size=batch_size, shuffle=True)
    collate_fn_train = MyCollateFn(X_train, y_train, group_train)
    dataloader_train = DataLoader(
        range(len(X_train)),
        batch_sampler=batch_sampler,  # 这里用 batch_sampler 替代 shuffle=True
        num_workers=48,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        collate_fn=collate_fn_train
    )
    print('len(dataloader_train)',len(dataloader_train),'cost time:',time.time()-t)
    # 3. 初始化模型和优化器
    input_size = X_train.shape[1]
    model = MLPmodel(input_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    min_loss = float('inf')
    no_improve_count = 0
    best_model_state = None
    print('margin',margin)
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
            outputs = model(batch_X).squeeze(-1)  # 保证输出与 batch_y 形状一致

            # 使用分组排序损失：只比较相同 candle_begin_time 下的样本
            loss = calc_loss(batch_y, outputs, batch_groups, margin=margin)
            if torch.isnan(loss):
                print("Loss is NaN! Stopping training.")
                return None, None, None
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(dataloader_train)
        elapsed_epoch = time.time() - start_epoch

        # Early Stopping 逻辑
        if epoch_loss < min_loss:
            min_loss = epoch_loss
            no_improve_count = 0
            best_model_state = model.state_dict()
        else:
            no_improve_count += 1

        if no_improve_count >= patience and epoch >= epochs:
            print(f"Early stopping at epoch {epoch+1-patience} with loss {epoch_loss:.4f}")
            break

        if epoch % 10 == 0:
            print(f"Epoch [{epoch}/{max_epochs}], Loss: {epoch_loss:.4f}, Time: {elapsed_epoch:.2f}s")

    # 5. 评估阶段：加载最佳模型并计算 MSE 与 Spearman 相关性
    model.load_state_dict(best_model_state)
    model.eval()
    with torch.no_grad():
        X_test_tensor = X_test_tensor.to(device)
        y_pred = model(X_test_tensor).squeeze(-1).cpu().numpy()
        y_test_numpy = y_test_tensor.cpu().numpy()

        loss = mean_squared_error(y_test_numpy, y_pred)
        print(f"Test primary ranking metric: {loss:.4f}")

        # 计算每个时间段内 Spearman 相关性（假设 test_data 中存在 candle_begin_time 列）
        test_data_copy = test_data.copy()
        test_data_copy['preds'] = y_pred
        # 注意：这里需要 groupby 依据 'candle_begin_time' 计算相关性
        hourly_corr = test_data_copy.groupby('candle_begin_time')[['preds', TARGET]]\
                                     .corr(method='spearman').unstack().iloc[:, 1]
        mean_corr = hourly_corr.mean()
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
        # if model is not None:
        #     model_path = os.path.join(args.save_dir, f"model_{split_name}.pth")
        #     torch.save(model.state_dict(), model_path)
        #     print(f"Saved model to {model_path}")
# def main(args):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")
#     if torch.cuda.is_available():
#         print(f"GPU Name: {torch.cuda.get_device_name(0)}")
#         print(f"CUDA Version: {torch.version.cuda}")
#         print(f"GPU Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
#         print(f"GPU Memory Cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")

#     # 数据加载和预处理
#     data, USE_COLS, TARGET = load_and_preprocess_data(args.data_path)
#     print(f"Number of features used: {len(USE_COLS)}")
#     test_data=data[pd.to_datetime(data['candle_begin_time']) >= pd.to_datetime('2025-01-01')].reset_index(drop=True)
#     train_data=data[pd.to_datetime(data['candle_begin_time']) < pd.to_datetime('2025-01-01')].reset_index(drop=True)
#     model, rmse, mean_corr = train_and_evaluate(train_data, test_data, USE_COLS, TARGET,
#                                                     lr=args.learning_rate,
#                                                     batch_size=args.batch_size,
#                                                     epochs=args.epochs,
#                                                     patience=args.patience,
#                                                     max_epochs=args.max_epochs,
#                                                     device=device,
#                                                     num_workers=args.num_workers)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep Learning Project for MLP Regression")
    parser.add_argument("--data_path", type=str, default="processed_data1.csv", help="Path to the CSV data file")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer")
    parser.add_argument("--batch_size", type=int, default=20000, help="Batch size for DataLoader")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs before early stopping can kick in")
    parser.add_argument("--patience", type=int, default=10, help="Patience for early stopping")
    parser.add_argument("--max_epochs", type=int, default=500, help="Maximum number of training epochs")
    parser.add_argument("--num_workers", type=int, default=48, help="Number of workers for DataLoader")
    parser.add_argument("--save_dir", type=str, default="./models", help="Directory to save the trained models")
    args = parser.parse_args()
    # os.makedirs(args.save_dir, exist_ok=True)
    main(args)