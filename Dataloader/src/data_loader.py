import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import math

class TimeSeriesDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class DataProcessor:
    def __init__(self, data_path, test_size=0.2, batch_size=32):
        self.data_path = data_path
        self.test_size = test_size
        self.batch_size = batch_size
        
        # 为不同物理特征创建单独的标准化器
        self.energy_scaler = StandardScaler()  # 能量标准化
        self.atomic_number_scaler = StandardScaler()  # 原子序数标准化
        self.mfp_scaler = StandardScaler()  # 自由程标准化
        
        # 为不同类型的输出创建单独的标准化器
        self.no_scatter_scaler = StandardScaler()  # 不考虑散射的累积因子
        self.scatter_scaler = StandardScaler()  # 考虑散射的累积因子
    
    def load_data(self):
        # 读取数据
        data = pd.read_csv(self.data_path, sep='\s+', header=None)
        
        # 根据物理意义分离特征
        energy = data.iloc[:, 0:1].values  # 能量
        atomic_numbers = data.iloc[:, [1,3,5,7]].values  # 四层材料的原子序数
        mfp_values = data.iloc[:, [2,4,6,8]].values  # 四层材料的自由程
        
        # 根据物理意义分离目标
        no_scatter_targets = data.iloc[:, 9:12].values  # 不考虑散射的累积因子
        scatter_targets = data.iloc[:, 12:15].values  # 考虑散射的累积因子
        
        # 分别对不同物理特征进行标准化
        energy_scaled = self.energy_scaler.fit_transform(energy)
        atomic_numbers_scaled = self.atomic_number_scaler.fit_transform(atomic_numbers)
        mfp_scaled = self.mfp_scaler.fit_transform(mfp_values)  # 自由程标准化
        
        # 分别对不同类型的输出进行标准化
        no_scatter_scaled = self.no_scatter_scaler.fit_transform(no_scatter_targets)
        scatter_scaled = self.scatter_scaler.fit_transform(scatter_targets)
        
        # 重新组合特征和目标
        features = np.column_stack((energy_scaled, 
                                   atomic_numbers_scaled[:, 0], mfp_scaled[:, 0],
                                   atomic_numbers_scaled[:, 1], mfp_scaled[:, 1],
                                   atomic_numbers_scaled[:, 2], mfp_scaled[:, 2],
                                   atomic_numbers_scaled[:, 3], mfp_scaled[:, 3]))
        targets = np.column_stack((no_scatter_scaled, scatter_scaled))

        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            features, targets,
            test_size=self.test_size,
            random_state=42
        )

        # 创建数据加载器
        train_dataset = TimeSeriesDataset(X_train, y_train)
        test_dataset = TimeSeriesDataset(X_test, y_test)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )

        return train_loader, test_loader

    def inverse_transform_targets(self, predictions):
        """将标准化的预测结果转换回原始尺度"""
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
            
        # 分离不同类型的预测结果
        no_scatter_preds = predictions[:, :3]
        scatter_preds = predictions[:, 3:]
        
        # 分别转换回原始尺度
        no_scatter_original = self.no_scatter_scaler.inverse_transform(no_scatter_preds)
        scatter_original = self.scatter_scaler.inverse_transform(scatter_preds)
        
        # 重新组合结果
        return np.column_stack((no_scatter_original, scatter_original))