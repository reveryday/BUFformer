import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import math

# 封装时间序列数据
class TimeSeriesDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)   #转换为pytorch格式
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class DataProcessor:
    def __init__(self, data_path, test_size=0.2, val_size=0.1, batch_size=32, use_robust_scaling=True, 
                 use_data_augmentation=True, augmentation_factor=0.2):
        self.data_path = data_path
        self.test_size = test_size
        self.val_size = val_size
        self.batch_size = batch_size
        self.use_robust_scaling = use_robust_scaling #鲁棒缩放
        self.use_data_augmentation = use_data_augmentation
        self.augmentation_factor = augmentation_factor
        
        # 为不同物理特征创建单独的标准化器
        if use_robust_scaling:
            # 使用鲁棒缩放，更好地处理异常值
            self.energy_scaler = RobustScaler()  # 能量标准化
            self.atomic_number_scaler = RobustScaler()  # 原子序数标准化
            self.mfp_scaler = RobustScaler()  # 自由程标准化
            
            # 为不同类型的输出创建单独的标准化器
            self.no_scatter_scaler = RobustScaler()  # 不考虑散射的累积因子
            self.scatter_scaler = RobustScaler()  # 考虑散射的累积因子
        else:
            # 使用标准缩放
            self.energy_scaler = StandardScaler()  # 能量标准化
            self.atomic_number_scaler = StandardScaler()  # 原子序数标准化
            self.mfp_scaler = StandardScaler()  # 自由程标准化
            
            # 为不同类型的输出创建单独的标准化器
            self.no_scatter_scaler = StandardScaler()  # 不考虑散射的累积因子
            self.scatter_scaler = StandardScaler()  # 考虑散射的累积因子
    
    def feature_engineering(self, data):
        """增加特征工程，创建物理特性相关的复合特征"""
        # 提取原始特征
        energy = data.iloc[:, 0:1].values  # 能量
        atomic_numbers = data.iloc[:, [1,3,5,7]].values  # 四层材料的原子序数
        mfp_values = data.iloc[:, [2,4,6,8]].values  # 四层材料的自由程
        
        # 创建新特征
        # 1. 能量与原子序数的乘积（物理意义：描述能量与材料相互作用的强度）
        energy_atomic_interaction = np.zeros((len(data), 4))
        for i in range(4):
            energy_atomic_interaction[:, i] = energy.flatten() * atomic_numbers[:, i]
        
        # 2. 累积自由程（物理意义：描述累积屏蔽效应）
        cumulative_mfp = np.zeros((len(data), 4))
        for i in range(4):
            if i == 0:
                cumulative_mfp[:, i] = mfp_values[:, i]
            else:
                cumulative_mfp[:, i] = cumulative_mfp[:, i-1] + mfp_values[:, i]
        
        # 3. 自由程与原子序数的比值（物理意义：描述材料对辐射的阻挡效率）
        mfp_atomic_ratio = np.zeros((len(data), 4))
        for i in range(4):
            # 避免除以零
            safe_atomic = np.maximum(atomic_numbers[:, i], 1)
            mfp_atomic_ratio[:, i] = mfp_values[:, i] / safe_atomic
        
        # 4. 自由程的平方（物理意义：考虑非线性衰减效应）
        mfp_squared = np.square(mfp_values)
        
        # 返回原始特征和工程特征
        return {
            'energy': energy,
            'atomic_numbers': atomic_numbers,
            'mfp_values': mfp_values,
            'energy_atomic_interaction': energy_atomic_interaction,
            'cumulative_mfp': cumulative_mfp,
            'mfp_atomic_ratio': mfp_atomic_ratio,
            'mfp_squared': mfp_squared
        }
    
    def data_augmentation(self, features_dict, targets):
        """数据增强，通过添加细微扰动生成更多训练样本"""
        if not self.use_data_augmentation:
            return features_dict, targets
        
        n_samples = targets.shape[0]
        n_augment = int(n_samples * self.augmentation_factor)
        
        # 随机选择样本进行增强
        indices = np.random.choice(n_samples, n_augment, replace=True)
        
        # 为每个特征创建增强版本
        augmented_data = {}
        for key, value in features_dict.items():
            # 原始数据
            original = value.copy()
            
            # 增强数据（添加细微随机扰动）
            augmented = original[indices].copy()
            
            # 根据不同特征类型添加不同程度的扰动
            if key == 'energy':
                noise = np.random.normal(0, 0.03, augmented.shape) * augmented
            elif key in ['atomic_numbers', 'mfp_values']:
                noise = np.random.normal(0, 0.05, augmented.shape) * augmented
            else:
                noise = np.random.normal(0, 0.1, augmented.shape) * augmented
            
            augmented += noise
            
            # 确保物理约束（如原子序数必须为正）
            if key == 'atomic_numbers':
                augmented = np.maximum(augmented, 1)
            
            # 合并原始和增强数据
            augmented_data[key] = np.vstack([original, augmented])
        
        # 增强目标值
        augmented_targets = targets[indices].copy()
        noise_targets = np.random.normal(0, 0.02, augmented_targets.shape) * augmented_targets
        augmented_targets += noise_targets
        
        # 确保物理约束（所有累积因子必须为正）
        augmented_targets = np.maximum(augmented_targets, 0)
        
        # 合并原始和增强目标
        combined_targets = np.vstack([targets, augmented_targets])
        
        return augmented_data, combined_targets
    
    def load_data(self):
        # 读取数据
        data = pd.read_csv(self.data_path, sep='\s+', header=None)
        
        # 应用特征工程
        features_dict = self.feature_engineering(data)
        
        # 根据物理意义分离目标
        no_scatter_targets = data.iloc[:, 9:12].values  # 不考虑散射的累积因子
        scatter_targets = data.iloc[:, 12:15].values  # 考虑散射的累积因子
        targets = np.column_stack((no_scatter_targets, scatter_targets))
        
        # 应用数据增强
        if self.use_data_augmentation:
            features_dict, targets = self.data_augmentation(features_dict, targets)
        
        # 标准化
        energy_scaled = self.energy_scaler.fit_transform(features_dict['energy'])
        atomic_numbers_scaled = self.atomic_number_scaler.fit_transform(features_dict['atomic_numbers'])
        mfp_scaled = self.mfp_scaler.fit_transform(features_dict['mfp_values'])
        
        # 标准化或归一化工程特征
        energy_atomic_scaled = self.energy_scaler.transform(features_dict['energy_atomic_interaction'])
        cumulative_mfp_scaled = self.mfp_scaler.transform(features_dict['cumulative_mfp'])
        mfp_atomic_ratio_scaled = StandardScaler().fit_transform(features_dict['mfp_atomic_ratio'])
        mfp_squared_scaled = StandardScaler().fit_transform(features_dict['mfp_squared'])
        
        # 标准化目标
        no_scatter_scaled = self.no_scatter_scaler.fit_transform(targets[:, :3])
        scatter_scaled = self.scatter_scaler.fit_transform(targets[:, 3:])
        
        # 组合所有特征
        # 原始特征
        features = np.column_stack((
            energy_scaled, 
            atomic_numbers_scaled[:, 0], mfp_scaled[:, 0],
            atomic_numbers_scaled[:, 1], mfp_scaled[:, 1],
            atomic_numbers_scaled[:, 2], mfp_scaled[:, 2],
            atomic_numbers_scaled[:, 3], mfp_scaled[:, 3],
            # 添加工程特征
            energy_atomic_scaled,
            cumulative_mfp_scaled,
            mfp_atomic_ratio_scaled,
            mfp_squared_scaled
        ))
        
        # 组合目标
        targets_scaled = np.column_stack((no_scatter_scaled, scatter_scaled))

        # 首先将数据分成训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            features, targets_scaled,
            test_size=self.test_size,
            random_state=42
        )
        
        # 从训练集中分出验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train,
            test_size=self.val_size / (1 - self.test_size),  # 调整比例
            random_state=42
        )

        # 创建数据加载器
        train_dataset = TimeSeriesDataset(X_train, y_train)
        val_dataset = TimeSeriesDataset(X_val, y_val)
        test_dataset = TimeSeriesDataset(X_test, y_test)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=4
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=4
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=4
        )
        
        # 保存特征数量，便于模型构建
        self.feature_dim = features.shape[1]

        return train_loader, val_loader, test_loader

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