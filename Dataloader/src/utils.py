import torch
import numpy as np
import matplotlib.pyplot as plt

def set_seed(seed):
    """设置随机种子以确保结果可重现"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def plot_feature_importance(model, feature_names=None):
    """可视化特征重要性（基于输入投影层的权重）"""
    if feature_names is None:
        feature_names = [f'特征 {i+1}' for i in range(model.input_projection.weight.shape[1])]
    
    # 获取输入投影层的权重
    weights = model.input_projection.weight.detach().cpu().numpy()
    
    # 计算每个特征的重要性（基于权重的绝对值平均）
    importance = np.mean(np.abs(weights), axis=0)
    
    # 创建特征重要性图
    plt.figure(figsize=(10, 6))
    plt.bar(feature_names, importance)
    plt.xlabel('输入特征')
    plt.ylabel('重要性分数')
    plt.title('特征重要性')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

def save_prediction_results(predictions, targets=None, file_path='prediction_results.csv'):
    """保存预测结果到CSV文件"""
    import pandas as pd
    
    # 创建结果DataFrame
    results = pd.DataFrame(predictions, columns=[f'预测_{i+1}' for i in range(predictions.shape[1])])
    
    # 如果有真实值，也保存
    if targets is not None:
        targets_df = pd.DataFrame(targets, columns=[f'真实_{i+1}' for i in range(targets.shape[1])])
        results = pd.concat([results, targets_df], axis=1)
    
    # 保存到CSV
    results.to_csv(file_path, index=False)
    print(f'预测结果已保存到 {file_path}')