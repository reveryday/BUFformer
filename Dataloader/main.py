import os
import argparse
import torch
import numpy as np
from src.data_loader import DataProcessor
from src.model import TransformerPredictor
from src.train import Trainer
from src.evaluate import Evaluator

def set_seed(seed):
    """设置随机种子以确保结果可复现"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='XSformer - 辐射屏蔽设计优化模型')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'evaluate', 'predict'],
                        help='运行模式：train（训练）, evaluate（评估）, predict（预测）')
    parser.add_argument('--data_path', type=str, default='data.txt',
                        help='数据文件路径')
    parser.add_argument('--model_path', type=str, default='best_model.pth',
                        help='模型文件路径')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='批次大小')
    parser.add_argument('--epochs', type=int, default=200,
                        help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='学习率')
    parser.add_argument('--d_model', type=int, default=128, 
                        help='模型维度')
    parser.add_argument('--nhead', type=int, default=8, 
                        help='注意力头数')
    parser.add_argument('--num_encoder_layers', type=int, default=4, 
                        help='编码器层数')
    parser.add_argument('--num_decoder_layers', type=int, default=4, 
                        help='解码器层数')
    parser.add_argument('--dim_feedforward', type=int, default=512, 
                        help='前馈网络维度')
    parser.add_argument('--dropout', type=float, default=0.1, 
                        help='Dropout比例')
    parser.add_argument('--seed', type=int, default=42, 
                        help='随机种子')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建结果目录
    os.makedirs('results', exist_ok=True)
    
    # 初始化数据处理器
    data_processor = DataProcessor(args.data_path, batch_size=args.batch_size)
    train_loader, test_loader = data_processor.load_data()
    
    # 初始化模型
    model = TransformerPredictor(
        input_dim=9,  # 能量(1) + 原子序数(4) + 自由程(4)
        output_dim=6, # 不考虑散射(3) + 考虑散射(3)
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout
    )
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    if args.mode == 'train':
        # 训练模式
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            data_processor=data_processor,
            learning_rate=args.learning_rate,
            num_epochs=args.epochs
        )
        trainer.train()
        print("训练完成，最佳模型已保存为 'best_model.pth'")
        
    elif args.mode in ['evaluate', 'predict']:
        # 加载预训练模型
        model.load_state_dict(torch.load(args.model_path))
        print(f"已加载模型: {args.model_path}")
    
    # 评估模型
    evaluator = Evaluator(model, test_loader, data_processor)
    
    if args.mode == 'evaluate':
        metrics, _ = evaluator.evaluate()
        print('\n评估结果：')
        for output_name, output_metrics in metrics.items():
            if output_name != 'average':
                print(f'\n{output_name}:')
                for metric_name, value in output_metrics.items():
                    print(f'{metric_name}: {value:.4f}')
        
        print('\n平均指标:')
        for metric_name, value in metrics['average'].items():
            print(f'{metric_name}: {value:.4f}')
                
    elif args.mode == 'predict':
        # 使用测试集的第一个批次进行预测示例
        features, _ = next(iter(test_loader))
        predictions = evaluator.predict(features[:5])
        
        # 获取原始特征数据
        features_np = features[:5].cpu().numpy()
        energy = data_processor.energy_scaler.inverse_transform(features_np[:, 0:1])
        atomic_numbers = data_processor.atomic_number_scaler.inverse_transform(features_np[:, [1,3,5,7]])
        mfp_values = data_processor.mfp_scaler.inverse_transform(features_np[:, [2,4,6,8]])
        
        print('\n预测结果示例：')
        output_names = ['注量累积因子', '照射量累积因子', '有效剂量累积因子']
        
        for i in range(5):
            print(f'\n样本 {i+1}:')
            print(f'  能量: {energy[i][0]:.2f} MeV')
            print(f'  材料1: 原子序数={atomic_numbers[i][0]:.0f}, 自由程={mfp_values[i][0]:.4f}')
            print(f'  材料2: 原子序数={atomic_numbers[i][1]:.0f}, 自由程={mfp_values[i][1]:.4f}')
            print(f'  材料3: 原子序数={atomic_numbers[i][2]:.0f}, 自由程={mfp_values[i][2]:.4f}')
            print(f'  材料4: 原子序数={atomic_numbers[i][3]:.0f}, 自由程={mfp_values[i][3]:.4f}')
            
            print('  预测结果:')
            for j in range(3):
                print(f'    不考虑散射-{output_names[j]}: {predictions[i][j]:.6f}')
            for j in range(3):
                print(f'    考虑散射-{output_names[j]}: {predictions[i][j+3]:.6f}')

if __name__ == '__main__':
    main()