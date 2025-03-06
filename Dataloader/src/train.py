import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau

class Trainer:
    def __init__(self, model, train_loader, test_loader, data_processor,
                 learning_rate=0.001, num_epochs=100):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.data_processor = data_processor
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # 定义损失函数和优化器
        self.criterion = nn.MSELoss()
        self.l1_loss = nn.L1Loss()  # 添加L1损失用于正则化
        
        # 为不同类型的累积因子定义权重
        self.no_scatter_weights = torch.tensor([1.0, 1.0, 1.2], device=self.device)  # 不考虑散射的三种累积因子权重
        self.scatter_weights = torch.tensor([1.2, 1.0, 1.2], device=self.device)    # 考虑散射的三种累积因子权重
        
        # 优化器和学习率调度器
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10)
        
        # 训练配置
        self.num_epochs = num_epochs
        self.train_losses = []
        self.val_losses = []
        
        # 物理约束参数
        self.physics_weight = 0.1  # 物理约束损失权重
        
    def physics_constraint_loss(self, outputs, features):
        """物理约束损失函数，确保预测结果符合辐射物理规律"""
        # 获取原始特征
        energy = features[:, 0:1]  # 能量
        
        # 1. 能量越高，累积因子应该越小（负相关）
        energy_constraint = torch.mean(torch.relu(torch.sum(outputs * energy, dim=1)))
        
        # 2. 不考虑散射的累积因子应小于考虑散射的累积因子
        no_scatter_outputs = outputs[:, :3]  # 不考虑散射的预测
        scatter_outputs = outputs[:, 3:]     # 考虑散射的预测
        scatter_constraint = torch.mean(torch.relu(no_scatter_outputs - scatter_outputs))
        
        # 3. 确保所有预测值为正
        positivity_constraint = torch.mean(torch.relu(-outputs))
        
        return energy_constraint + scatter_constraint + positivity_constraint
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        for batch_features, batch_targets in tqdm(self.train_loader, desc='Training'):
            batch_features = batch_features.to(self.device)
            batch_targets = batch_targets.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(batch_features)
            
            # 计算加权损失
            # 分离不同类型的预测和目标
            no_scatter_outputs = outputs[:, :3]  # 不考虑散射的预测
            scatter_outputs = outputs[:, 3:]     # 考虑散射的预测
            no_scatter_targets = batch_targets[:, :3]  # 不考虑散射的目标
            scatter_targets = batch_targets[:, 3:]     # 考虑散射的目标
            
            # 计算不同部分的损失并加权
            no_scatter_mse = torch.mean(self.no_scatter_weights * torch.pow(no_scatter_outputs - no_scatter_targets, 2))
            scatter_mse = torch.mean(self.scatter_weights * torch.pow(scatter_outputs - scatter_targets, 2))
            
            # 添加L1损失以提高鲁棒性
            no_scatter_l1 = self.l1_loss(no_scatter_outputs, no_scatter_targets)
            scatter_l1 = self.l1_loss(scatter_outputs, scatter_targets)
            
            # 添加物理约束损失
            physics_loss = self.physics_constraint_loss(outputs, batch_features)
            
            # 总损失 = MSE损失 + L1正则化 + 物理约束
            loss = (no_scatter_mse + scatter_mse) + 0.1 * (no_scatter_l1 + scatter_l1) + self.physics_weight * physics_loss
            
            # 反向传播和优化
            loss.backward()
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(self.train_loader)
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch_features, batch_targets in self.test_loader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                # 前向传播
                outputs = self.model(batch_features)
                
                # 计算加权损失
                # 分离不同类型的预测和目标
                no_scatter_outputs = outputs[:, :3]  # 不考虑散射的预测
                scatter_outputs = outputs[:, 3:]     # 考虑散射的预测
                no_scatter_targets = batch_targets[:, :3]  # 不考虑散射的目标
                scatter_targets = batch_targets[:, 3:]     # 考虑散射的目标
                
                # 计算不同部分的损失并加权
                no_scatter_mse = torch.mean(self.no_scatter_weights * torch.pow(no_scatter_outputs - no_scatter_targets, 2))
                scatter_mse = torch.mean(self.scatter_weights * torch.pow(scatter_outputs - scatter_targets, 2))
                
                # 总损失
                loss = no_scatter_mse + scatter_mse
                total_loss += loss.item()
        
        return total_loss / len(self.test_loader)
    
    def train(self):
        print(f'Training on {self.device}')
        best_val_loss = float('inf')
        patience = 20  # 早停耐心值
        counter = 0    # 早停计数器
        
        for epoch in range(self.num_epochs):
            # 训练一个epoch
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # 验证
            val_loss = self.validate()
            self.val_losses.append(val_loss)
            
            # 更新学习率
            self.scheduler.step(val_loss)
            
            # 打印进度
            print(f'Epoch [{epoch+1}/{self.num_epochs}] '
                  f'Train Loss: {train_loss:.4f} '
                  f'Val Loss: {val_loss:.4f} '
                  f'LR: {self.optimizer.param_groups[0]["lr"]:.6f}')
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_model.pth')
                counter = 0  # 重置早停计数器
            else:
                counter += 1  # 增加早停计数器
            
            # 早停机制
            if counter >= patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break
        
        # 绘制损失曲线
        self.plot_losses()
    
    def plot_losses(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses')
        plt.legend()
        plt.savefig('training_loss.png')
        plt.close()