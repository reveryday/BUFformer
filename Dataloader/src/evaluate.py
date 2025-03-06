import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
import seaborn as sns
from torch.cuda.amp import autocast
import pandas as pd
import os

class Evaluator:
    def __init__(self, model, test_loader, data_processor):
        self.model = model
        self.test_loader = test_loader
        self.data_processor = data_processor
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # 创建结果目录
        os.makedirs('results', exist_ok=True)
        os.makedirs('results/figures', exist_ok=True)
        
    def evaluate(self, use_amp=True):
        """评估模型性能"""
        self.model.eval()
        all_targets = []
        all_predictions = []
        all_features = []
        
        with torch.no_grad():
            for batch_features, batch_targets in self.test_loader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                # 使用混合精度推理
                with autocast(enabled=use_amp and torch.cuda.is_available()):
                    # 前向传播
                    outputs = self.model(batch_features)
                
                # 收集结果
                all_targets.append(batch_targets.cpu().numpy())
                all_predictions.append(outputs.cpu().numpy())
                all_features.append(batch_features.cpu().numpy())
        
        # 合并批次结果
        targets = np.vstack(all_targets)
        predictions = np.vstack(all_predictions)
        features = np.vstack(all_features)
        
        # 转换回原始尺度
        targets_original = self.data_processor.inverse_transform_targets(targets)
        predictions_original = self.data_processor.inverse_transform_targets(predictions)
        
        # 计算评估指标
        metrics = self._calculate_metrics(targets_original, predictions_original)
        
        # 保存结果到CSV
        self._save_predictions(features, targets_original, predictions_original)
        
        # 可视化结果
        self._visualize_predictions(targets_original, predictions_original)
        
        # 分析预测结果
        self._analyze_predictions_by_energy(features, targets_original, predictions_original)
        self._analyze_predictions_by_material(features, targets_original, predictions_original)
        
        return metrics, predictions_original
    
    def _calculate_metrics(self, targets, predictions):
        """计算评估指标"""
        metrics = {}
        
        # 计算每个输出维度的指标
        for i in range(targets.shape[1]):
            # 基本指标
            mse = mean_squared_error(targets[:, i], predictions[:, i])
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(targets[:, i], predictions[:, i])
            r2 = r2_score(targets[:, i], predictions[:, i])
            evs = explained_variance_score(targets[:, i], predictions[:, i])
            
            # 相对指标
            mean_target = np.mean(np.abs(targets[:, i]))
            rel_error = mae / mean_target * 100 if mean_target > 0 else float('nan')
            
            # 计算不同误差级别的百分比
            abs_errors = np.abs(targets[:, i] - predictions[:, i])
            pct_within_5 = np.mean(abs_errors <= 0.05 * np.abs(targets[:, i])) * 100
            pct_within_10 = np.mean(abs_errors <= 0.1 * np.abs(targets[:, i])) * 100
            pct_within_20 = np.mean(abs_errors <= 0.2 * np.abs(targets[:, i])) * 100
            
            # 根据物理意义命名输出
            output_names = ['注量累积因子', '照射量累积因子', '有效剂量累积因子']
            scatter_prefix = '不考虑散射' if i < 3 else '考虑散射'
            current_name = f'{scatter_prefix}-{output_names[i % 3]}'
            
            metrics[current_name] = {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2,
                '解释方差': evs,
                '相对误差(%)': rel_error,
                '5%误差内比例(%)': pct_within_5,
                '10%误差内比例(%)': pct_within_10,
                '20%误差内比例(%)': pct_within_20
            }
        
        # 计算平均指标
        avg_metrics = {}
        for metric in metrics[list(metrics.keys())[0]].keys():
            avg_metrics[metric] = np.mean([metrics[name][metric] for name in metrics if name != 'average'])
        
        metrics['平均'] = avg_metrics
        
        # 生成评估报告
        self._generate_evaluation_report(metrics)
        
        return metrics
    
    def _generate_evaluation_report(self, metrics):
        """生成详细的评估报告"""
        # 创建表格数据
        data = []
        for output_name, output_metrics in metrics.items():
            row = {'输出': output_name}
            row.update(output_metrics)
            data.append(row)
        
        # 创建DataFrame
        df = pd.DataFrame(data)
        
        # 保存为CSV
        df.to_csv('results/evaluation_metrics.csv', index=False)
        
        # 保存为Markdown表格
        with open('results/evaluation_report.md', 'w') as f:
            f.write('# 模型评估报告\n\n')
            f.write('## 性能指标\n\n')
            markdown_table = df.to_markdown(index=False, floatfmt='.4f')
            f.write(markdown_table)
            f.write('\n\n')
            
            # 添加指标解释
            f.write('## 指标说明\n\n')
            f.write('- **MSE**: 均方误差，越小越好\n')
            f.write('- **RMSE**: 均方根误差，越小越好\n')
            f.write('- **MAE**: 平均绝对误差，越小越好\n')
            f.write('- **R2**: 决定系数，越接近1越好，表示模型解释的方差比例\n')
            f.write('- **解释方差**: 解释方差分数，越接近1越好\n')
            f.write('- **相对误差(%)**: MAE除以目标均值的百分比，越小越好\n')
            f.write('- **5%/10%/20%误差内比例(%)**: 预测值在真实值的5%/10%/20%误差范围内的样本百分比，越高越好\n')
    
    def _save_predictions(self, features, targets, predictions):
        """保存预测结果到CSV文件"""
        # 提取原始特征
        energy = self.data_processor.energy_scaler.inverse_transform(features[:, 0:1])
        atomic_numbers = self.data_processor.atomic_number_scaler.inverse_transform(features[:, [1,3,5,7]])
        mfp_values = self.data_processor.mfp_scaler.inverse_transform(features[:, [2,4,6,8]])
        
        # 创建DataFrame
        results = pd.DataFrame()
        
        # 添加输入特征
        results['能量'] = energy.flatten()
        for i in range(4):
            results[f'层{i+1}_原子序数'] = atomic_numbers[:, i]
            results[f'层{i+1}_厚度'] = mfp_values[:, i]
            
        # 添加预测和真实值
        output_names = ['注量累积因子', '照射量累积因子', '有效剂量累积因子']
        for i in range(6):
            scatter_prefix = '不考虑散射' if i < 3 else '考虑散射'
            current_name = f'{scatter_prefix}_{output_names[i % 3]}'
            
            results[f'预测_{current_name}'] = predictions[:, i]
            results[f'真实_{current_name}'] = targets[:, i]
            results[f'误差_{current_name}'] = predictions[:, i] - targets[:, i]
            results[f'相对误差_{current_name}'] = np.abs(predictions[:, i] - targets[:, i]) / np.abs(targets[:, i]) * 100
        
        # 保存到CSV
        results.to_csv('results/prediction_results.csv', index=False)
        
        # 保存一个样本子集用于快速查看
        sample_indices = np.random.choice(len(results), min(100, len(results)), replace=False)
        results.iloc[sample_indices].to_csv('results/prediction_samples.csv', index=False)
    
    def _visualize_predictions(self, targets, predictions, num_samples=100):
        """可视化预测结果"""
        # 随机选择样本进行可视化
        indices = np.random.choice(len(targets), min(num_samples, len(targets)), replace=False)
        
        # 设置风格
        sns.set(style="whitegrid")
        
        # 为每个输出维度创建一个图
        for i in range(targets.shape[1]):
            plt.figure(figsize=(12, 10))
            
            # 绘制真实值和预测值的散点图
            scatter = plt.scatter(targets[indices, i], predictions[indices, i], 
                                c=np.abs(predictions[indices, i] - targets[indices, i]), 
                                cmap='viridis', alpha=0.7, s=80)
            
            # 添加对角线（完美预测线）
            min_val = min(targets[:, i].min(), predictions[:, i].min())
            max_val = max(targets[:, i].max(), predictions[:, i].max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
            
            # 添加颜色条，显示误差大小
            plt.colorbar(scatter, label='绝对误差')
            
            # 根据物理意义设置标题和标签
            output_names = ['注量累积因子', '照射量累积因子', '有效剂量累积因子']
            scatter_prefix = '不考虑散射' if i < 3 else '考虑散射'
            current_name = f'{scatter_prefix}-{output_names[i % 3]}'
            
            # 计算并显示指标
            mae = mean_absolute_error(targets[:, i], predictions[:, i])
            r2 = r2_score(targets[:, i], predictions[:, i])
            
            plt.title(f'{current_name}预测结果对比\nMAE: {mae:.4f}, R²: {r2:.4f}', fontsize=16)
            plt.xlabel('真实值', fontsize=14)
            plt.ylabel('预测值', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # 保存图像
            plt.savefig(f'results/figures/output_{i+1}_predictions.png', dpi=300)
            plt.close()
        
        # 创建一个汇总图，显示所有输出维度的预测误差分布
        plt.figure(figsize=(15, 10))
        
        for i in range(targets.shape[1]):
            errors = predictions[:, i] - targets[:, i]
            plt.subplot(2, 3, i+1)
            
            # 使用KDE图显示误差分布
            sns.histplot(errors, kde=True, bins=30)
            
            # 添加垂直线表示零误差
            plt.axvline(x=0, color='r', linestyle='--')
            
            # 根据物理意义设置误差分布图的标题和标签
            output_names = ['注量累积因子', '照射量累积因子', '有效剂量累积因子']
            scatter_prefix = '不考虑散射' if i < 3 else '考虑散射'
            current_name = f'{scatter_prefix}-{output_names[i % 3]}'
            
            plt.title(f'{current_name}误差分布', fontsize=12)
            plt.xlabel('预测误差')
            plt.ylabel('频数')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/figures/prediction_errors.png', dpi=300)
        plt.close()
        
        # 创建热图，显示各输出之间的相关性
        plt.figure(figsize=(14, 12))
        
        # 组合预测和真实值
        combined_data = np.column_stack((targets, predictions))
        
        # 创建列名
        output_names = ['注量累积因子', '照射量累积因子', '有效剂量累积因子']
        column_names = []
        for prefix in ['真实值_不考虑散射_', '真实值_考虑散射_', '预测值_不考虑散射_', '预测值_考虑散射_']:
            for name in output_names:
                column_names.append(f"{prefix}{name}")
        
        # 计算相关矩阵
        corr_matrix = np.corrcoef(combined_data.T)
        
        # 绘制热图
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
                   xticklabels=column_names, yticklabels=column_names)
        plt.title("输出变量相关性矩阵", fontsize=16)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('results/figures/correlation_matrix.png', dpi=300)
        plt.close()
    
    def _analyze_predictions_by_energy(self, features, targets, predictions):
        """按能量分析预测性能"""
        # 提取能量值
        energy = self.data_processor.energy_scaler.inverse_transform(features[:, 0:1]).flatten()
        
        # 设置能量区间
        energy_bins = np.linspace(energy.min(), energy.max(), 6)
        energy_labels = [f'{energy_bins[i]:.2f}-{energy_bins[i+1]:.2f}' for i in range(len(energy_bins)-1)]
        
        # 将能量分组
        energy_groups = np.digitize(energy, energy_bins[1:-1])
        
        # 为每个输出创建图表
        for out_idx in range(targets.shape[1]):
            # 创建数据框
            df = pd.DataFrame({
                '能量区间': [energy_labels[group] for group in energy_groups],
                '真实值': targets[:, out_idx],
                '预测值': predictions[:, out_idx],
                '绝对误差': np.abs(predictions[:, out_idx] - targets[:, out_idx]),
                '相对误差': np.abs(predictions[:, out_idx] - targets[:, out_idx]) / np.abs(targets[:, out_idx]) * 100
            })
            
            # 按能量区间分组计算均值
            grouped = df.groupby('能量区间').mean()
            
            # 输出名称
            output_names = ['注量累积因子', '照射量累积因子', '有效剂量累积因子']
            scatter_prefix = '不考虑散射' if out_idx < 3 else '考虑散射'
            current_name = f'{scatter_prefix}-{output_names[out_idx % 3]}'
            
            # 绘制按能量区间的误差条形图
            plt.figure(figsize=(12, 8))
            ax = grouped[['绝对误差', '相对误差']].plot(kind='bar', secondary_y='相对误差')
            plt.title(f'各能量区间的{current_name}预测误差', fontsize=14)
            ax.set_xlabel('能量区间 (MeV)', fontsize=12)
            ax.set_ylabel('绝对误差', fontsize=12)
            ax.right_ax.set_ylabel('相对误差 (%)', fontsize=12)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f'results/figures/energy_error_{out_idx}.png', dpi=300)
            plt.close()
            
            # 绘制能量与预测/真实值的散点图
            plt.figure(figsize=(12, 8))
            plt.scatter(energy, targets[:, out_idx], alpha=0.5, label='真实值')
            plt.scatter(energy, predictions[:, out_idx], alpha=0.5, label='预测值')
            
            # 添加趋势线
            z_true = np.polyfit(energy, targets[:, out_idx], 2)
            p_true = np.poly1d(z_true)
            z_pred = np.polyfit(energy, predictions[:, out_idx], 2)
            p_pred = np.poly1d(z_pred)
            
            x_range = np.linspace(energy.min(), energy.max(), 100)
            plt.plot(x_range, p_true(x_range), 'r--', linewidth=2, label='真实趋势')
            plt.plot(x_range, p_pred(x_range), 'g--', linewidth=2, label='预测趋势')
            
            plt.title(f'能量与{current_name}的关系', fontsize=14)
            plt.xlabel('能量 (MeV)', fontsize=12)
            plt.ylabel(current_name, fontsize=12)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'results/figures/energy_vs_{out_idx}.png', dpi=300)
            plt.close()
    
    def _analyze_predictions_by_material(self, features, targets, predictions):
        """按材料原子序数分析预测性能"""
        # 提取原子序数
        atomic_numbers = self.data_processor.atomic_number_scaler.inverse_transform(features[:, [1,3,5,7]])
        
        # 分析第一层材料
        layer1_z = atomic_numbers[:, 0]
        
        # 找出主要的材料原子序数(取最常见的5种)
        unique_z, counts = np.unique(layer1_z, return_counts=True)
        top_z = unique_z[np.argsort(counts)[-5:]]
        
        # 为每个输出创建箱线图
        for out_idx in range(targets.shape[1]):
            plt.figure(figsize=(14, 8))
            
            data_to_plot = []
            labels = []
            
            for z in top_z:
                mask = (layer1_z == z)
                rel_errors = np.abs(predictions[mask, out_idx] - targets[mask, out_idx]) / np.abs(targets[mask, out_idx]) * 100
                data_to_plot.append(rel_errors)
                labels.append(f'Z={int(z)}')
            
            # 箱线图
            plt.boxplot(data_to_plot, labels=labels)
            
            # 输出名称
            output_names = ['注量累积因子', '照射量累积因子', '有效剂量累积因子']
            scatter_prefix = '不考虑散射' if out_idx < 3 else '考虑散射'
            current_name = f'{scatter_prefix}-{output_names[out_idx % 3]}'
            
            plt.title(f'不同材料的{current_name}预测相对误差', fontsize=14)
            plt.ylabel('相对误差 (%)', fontsize=12)
            plt.xlabel('第一层材料原子序数', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            plt.savefig(f'results/figures/material_error_{out_idx}.png', dpi=300)
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
            with autocast(enabled=torch.cuda.is_available()):
                predictions = self.model(features)
        
        # 转换回原始尺度
        predictions_np = predictions.cpu().numpy()
        predictions_original = self.data_processor.inverse_transform_targets(predictions_np)
        
        return predictions_original