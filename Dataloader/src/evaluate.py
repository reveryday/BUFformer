import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class Evaluator:
    def __init__(self, model, test_loader, data_processor):
        self.model = model
        self.test_loader = test_loader
        self.data_processor = data_processor
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def evaluate(self):
        """评估模型性能"""
        self.model.eval()
        all_targets = []
        all_predictions = []
        
        with torch.no_grad():
            for batch_features, batch_targets in self.test_loader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                # 前向传播
                outputs = self.model(batch_features)
                
                # 收集结果
                all_targets.append(batch_targets.cpu().numpy())
                all_predictions.append(outputs.cpu().numpy())
        
        # 合并批次结果
        targets = np.vstack(all_targets)
        predictions = np.vstack(all_predictions)
        
        # 转换回原始尺度
        targets_original = self.data_processor.inverse_transform_targets(targets)
        predictions_original = self.data_processor.inverse_transform_targets(predictions)
        
        # 计算评估指标
        metrics = self._calculate_metrics(targets_original, predictions_original)
        
        # 可视化结果
        self._visualize_predictions(targets_original, predictions_original)
        
        return metrics, predictions_original
    
    def _calculate_metrics(self, targets, predictions):
        """计算评估指标"""
        metrics = {}
        
        # 计算每个输出维度的指标
        for i in range(targets.shape[1]):
            mse = mean_squared_error(targets[:, i], predictions[:, i])
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(targets[:, i], predictions[:, i])
            r2 = r2_score(targets[:, i], predictions[:, i])
            
            # 根据物理意义命名输出
            output_names = ['注量累积因子', '照射量累积因子', '有效剂量累积因子']
            scatter_prefix = '不考虑散射' if i < 3 else '考虑散射'
            current_name = f'{scatter_prefix}-{output_names[i % 3]}'
            
            metrics[current_name] = {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2,
                '相对误差': mae / np.mean(np.abs(targets[:, i])) * 100  # 添加相对误差指标
            }
        
        # 计算平均指标
        avg_mse = np.mean([metrics[name]['MSE'] for name in metrics if name != 'average'])
        avg_rmse = np.mean([metrics[name]['RMSE'] for name in metrics if name != 'average'])
        avg_mae = np.mean([metrics[name]['MAE'] for name in metrics if name != 'average'])
        avg_r2 = np.mean([metrics[name]['R2'] for name in metrics if name != 'average'])
        avg_rel_error = np.mean([metrics[name]['相对误差'] for name in metrics if name != 'average'])
        
        metrics['average'] = {
            'MSE': avg_mse,
            'RMSE': avg_rmse,
            'MAE': avg_mae,
            'R2': avg_r2,
            '相对误差': avg_rel_error
        }
        
        return metrics
    
    def _visualize_predictions(self, targets, predictions, num_samples=100):
        """可视化预测结果"""
        # 随机选择样本进行可视化
        indices = np.random.choice(len(targets), min(num_samples, len(targets)), replace=False)
        
        # 为每个输出维度创建一个图
        for i in range(targets.shape[1]):
            plt.figure(figsize=(12, 6))
            
            # 绘制真实值和预测值
            plt.scatter(targets[indices, i], predictions[indices, i], alpha=0.6)
            
            # 添加对角线（完美预测线）
            min_val = min(targets[:, i].min(), predictions[:, i].min())
            max_val = max(targets[:, i].max(), predictions[:, i].max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--')
            
            # 添加标题和标签
            # 根据物理意义设置标题和标签
            output_names = ['注量累积因子', '照射量累积因子', '有效剂量累积因子']
            scatter_prefix = '不考虑散射' if i < 3 else '考虑散射'
            current_name = f'{scatter_prefix}-{output_names[i % 3]}'
            
            plt.title(f'{current_name}预测结果对比')
            plt.xlabel('真实值')
            plt.ylabel('预测值')
            plt.grid(True, alpha=0.3)
            
            # 保存图像
            plt.savefig(f'output_{i+1}_predictions.png')
            plt.close()
        
        # 创建一个汇总图，显示所有输出维度的预测误差分布
        plt.figure(figsize=(14, 8))
        
        for i in range(targets.shape[1]):
            errors = predictions[:, i] - targets[:, i]
            plt.subplot(2, 3, i+1)
            plt.hist(errors, bins=30, alpha=0.7)
            # 根据物理意义设置误差分布图的标题和标签
            output_names = ['注量累积因子', '照射量累积因子', '有效剂量累积因子']
            scatter_prefix = '不考虑散射' if i < 3 else '考虑散射'
            current_name = f'{scatter_prefix}-{output_names[i % 3]}'
            
            plt.title(f'{current_name}误差分布')
            plt.xlabel('预测误差')
            plt.ylabel('频数')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('prediction_errors.png')
        plt.close()
        
        # 添加物理特性分析图
        self._analyze_physics_relationships(targets, predictions)
    
    def _analyze_physics_relationships(self, targets, predictions):
        """分析物理特性关系"""
        # 获取原始特征数据
        all_features = []
        with torch.no_grad():
            for batch_features, _ in self.test_loader:
                all_features.append(batch_features.cpu().numpy())
        
        features = np.vstack(all_features)
        
        # 分析能量与累积因子的关系
        energy = self.data_processor.energy_scaler.inverse_transform(features[:, 0:1])
        
        plt.figure(figsize=(15, 10))
        
        # 分析能量与各累积因子的关系
        for i in range(targets.shape[1]):
            plt.subplot(2, 3, i+1)
            
            # 绘制散点图
            plt.scatter(energy, targets[:, i], alpha=0.5, label='真实值')
            plt.scatter(energy, predictions[:, i], alpha=0.5, label='预测值')
            
            # 添加趋势线
            z_true = np.polyfit(energy.flatten(), targets[:, i], 1)
            p_true = np.poly1d(z_true)
            z_pred = np.polyfit(energy.flatten(), predictions[:, i], 1)
            p_pred = np.poly1d(z_pred)
            
            x_range = np.linspace(energy.min(), energy.max(), 100)
            plt.plot(x_range, p_true(x_range), 'r--', label='真实趋势')
            plt.plot(x_range, p_pred(x_range), 'g--', label='预测趋势')
            
            # 根据物理意义设置标题和标签
            output_names = ['注量累积因子', '照射量累积因子', '有效剂量累积因子']
            scatter_prefix = '不考虑散射' if i < 3 else '考虑散射'
            current_name = f'{scatter_prefix}-{output_names[i % 3]}'
            
            plt.title(f'能量与{current_name}关系')
            plt.xlabel('能量 (MeV)')
            plt.ylabel(current_name)
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('energy_vs_factors.png')
        plt.close()
    
    def predict(self, features):
        """使用模型进行预测"""
        self.model.eval()
        
        # 确保输入是张量
        if not isinstance(features, torch.Tensor):
            features = torch.FloatTensor(features).to(self.device)
        else:
            features = features.to(self.device)
        
        # 添加批次维度（如果需要）
        if len(features.shape) == 1:
            features = features.unsqueeze(0)
        
        # 前向传播
        with torch.no_grad():
            predictions = self.model(features)
        
        # 转换回原始尺度
        predictions_np = predictions.cpu().numpy()
        predictions_original = self.data_processor.inverse_transform_targets(predictions_np)
        
        return predictions_original