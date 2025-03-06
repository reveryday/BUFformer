import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
import os
import time
from torch.cuda.amp import autocast, GradScaler

class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, data_processor,
                 learning_rate=0.001, weight_decay=1e-5, num_epochs=200, 
                 use_amp=True, grad_accumulation_steps=1, warmup_epochs=5):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.data_processor = data_processor
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # 训练配置
        self.num_epochs = num_epochs
        self.use_amp = use_amp and torch.cuda.is_available()  # 只在GPU上使用混合精度
        self.grad_accumulation_steps = grad_accumulation_steps
        self.warmup_epochs = warmup_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # 定义损失函数
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.SmoothL1Loss()  # 平滑L1损失，对异常值更鲁棒
        
        # 为不同类型的累积因子定义权重
        self.no_scatter_weights = torch.tensor([1.2, 1.0, 1.5], device=self.device)  # 不考虑散射的三种累积因子权重
        self.scatter_weights = torch.tensor([1.5, 1.0, 1.5], device=self.device)    # 考虑散射的三种累积因子权重
        
        # 创建优化器
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # 学习率调度器
        steps_per_epoch = len(train_loader) // grad_accumulation_steps
        self.lr_scheduler = OneCycleLR(
            self.optimizer,
            max_lr=learning_rate,
            steps_per_epoch=steps_per_epoch,
            epochs=num_epochs,
            pct_start=warmup_epochs/num_epochs,  # 预热阶段占比
            anneal_strategy='cos',
            div_factor=25.0,  # 初始学习率 = max_lr/div_factor
            final_div_factor=1000.0  # 最终学习率 = max_lr/(div_factor*final_div_factor)
        )
        
        # 混合精度训练
        self.scaler = GradScaler(enabled=self.use_amp)
        
        # 训练过程记录
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        
        # 物理约束参数
        self.physics_weight = 0.2  # 物理约束损失权重
        
        # 创建检查点目录
        os.makedirs('checkpoints', exist_ok=True)
        
    def physics_constraint_loss(self, outputs, features):
        """物理约束损失函数，确保预测结果符合辐射物理规律"""
        # 获取原始特征
        energy = features[:, 0:1]  # 能量
        
        # 1. 能量越高，累积因子应该越小（负相关）
        energy_constraint = torch.mean(torch.relu(outputs * energy))
        
        # 2. 不考虑散射的累积因子应小于考虑散射的累积因子
        no_scatter_outputs = outputs[:, :3]  # 不考虑散射的预测
        scatter_outputs = outputs[:, 3:]     # 考虑散射的预测
        scatter_constraint = torch.mean(torch.relu(no_scatter_outputs - scatter_outputs))
        
        # 3. 确保所有预测值为正
        positivity_constraint = torch.mean(torch.relu(-outputs))
        
        # 4. 不同层累积效应约束（利用特征中的自由程或原子序数信息）
        mfp_values = features[:, [2,4,6,8]]  # 自由程
        total_mfp = torch.sum(mfp_values, dim=1, keepdim=True)
        attenuation_constraint = torch.mean(torch.relu(outputs - torch.exp(-total_mfp)))
        
        return energy_constraint + scatter_constraint + positivity_constraint + attenuation_constraint
        
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        steps = len(self.train_loader)
        
        # 进度条
        progress_bar = tqdm(enumerate(self.train_loader), total=steps, 
                           desc=f'Epoch {epoch+1}/{self.num_epochs}')
        
        running_loss = 0
        self.optimizer.zero_grad()
        
        for step, (batch_features, batch_targets) in progress_bar:
            batch_features = batch_features.to(self.device)
            batch_targets = batch_targets.to(self.device)
            
            # 使用混合精度训练
            with autocast(enabled=self.use_amp):
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
                
                # 添加L1损失以提高鲁棒性
                no_scatter_l1 = self.l1_loss(no_scatter_outputs, no_scatter_targets)
                scatter_l1 = self.l1_loss(scatter_outputs, scatter_targets)
                
                # 添加物理约束损失
                physics_loss = self.physics_constraint_loss(outputs, batch_features)
                
                # 总损失 = MSE损失 + L1正则化 + 物理约束
                loss = (no_scatter_mse + scatter_mse) + 0.2 * (no_scatter_l1 + scatter_l1) + self.physics_weight * physics_loss
                loss = loss / self.grad_accumulation_steps  # 归一化损失以进行梯度累积
            
            # 使用混合精度反向传播
            self.scaler.scale(loss).backward()
            running_loss += loss.item() * self.grad_accumulation_steps
            
            # 梯度累积更新
            if (step + 1) % self.grad_accumulation_steps == 0 or (step + 1) == steps:
                # 梯度裁剪，防止梯度爆炸
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # 优化器步进
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                
                # 更新学习率
                self.lr_scheduler.step()
                
                # 更新进度条
                progress_bar.set_postfix({
                    'loss': running_loss / self.grad_accumulation_steps, 
                    'lr': self.optimizer.param_groups[0]['lr']
                })
                
                total_loss += running_loss
                running_loss = 0
                
        self.learning_rates.append(self.optimizer.param_groups[0]['lr'])
        return total_loss / steps
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch_features, batch_targets in self.val_loader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                # 使用混合精度（仅用于推理，不影响性能）
                with autocast(enabled=self.use_amp):
                    # 前向传播
                    outputs = self.model(batch_features)
                    
                    # 计算加权损失
                    # 分离不同类型的预测和目标
                    no_scatter_outputs = outputs[:, :3]  # 不考虑散射的预测
                    scatter_outputs = outputs[:, 3:]     # 考虑散射的预测
                    no_scatter_targets = batch_targets[:, :3]  # 不考虑散射的目标
                    scatter_targets = batch_targets[:, 3:]     # 考虑散射的目标
                    
                    # 计算不同部分的损失
                    no_scatter_mse = torch.mean(self.no_scatter_weights * torch.pow(no_scatter_outputs - no_scatter_targets, 2))
                    scatter_mse = torch.mean(self.scatter_weights * torch.pow(scatter_outputs - scatter_targets, 2))
                    
                    # 添加L1损失
                    no_scatter_l1 = self.l1_loss(no_scatter_outputs, no_scatter_targets)
                    scatter_l1 = self.l1_loss(scatter_outputs, scatter_targets)
                    
                    # 总损失 (验证时不需要物理约束损失)
                    loss = (no_scatter_mse + scatter_mse) + 0.2 * (no_scatter_l1 + scatter_l1)
                
                total_loss += loss.item()
        
        return total_loss / len(self.val_loader)
    
    def train(self):
        print(f'使用设备: {self.device}')
        print(f'混合精度训练: {"启用" if self.use_amp else "禁用"}')
        print(f'梯度累积步数: {self.grad_accumulation_steps}')
        print(f'学习率预热轮数: {self.warmup_epochs}')
        print(f'开始训练...')
        
        start_time = time.time()
        best_val_loss = float('inf')
        patience = 20  # 早停耐心值
        counter = 0    # 早停计数器
        
        for epoch in range(self.num_epochs):
            epoch_start = time.time()
            
            # 训练一个epoch
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            
            # 验证
            val_loss = self.validate()
            self.val_losses.append(val_loss)
            
            epoch_time = time.time() - epoch_start
            
            # 打印进度
            print(f'Epoch [{epoch+1}/{self.num_epochs}] '
                  f'Train Loss: {train_loss:.6f} '
                  f'Val Loss: {val_loss:.6f} '
                  f'LR: {self.optimizer.param_groups[0]["lr"]:.6f} '
                  f'Time: {epoch_time:.1f}s')
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_model.pth')
                print(f'✓ 保存最佳模型 (验证损失: {val_loss:.6f})')
                counter = 0  # 重置早停计数器
            else:
                counter += 1  # 增加早停计数器
            
            # 定期保存检查点
            if (epoch + 1) % 10 == 0:
                checkpoint_path = f'checkpoints/model_epoch_{epoch+1}.pth'
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.lr_scheduler.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, checkpoint_path)
            
            # 早停机制
            if counter >= patience:
                print(f'提前停止训练 (在 {epoch+1} 轮后，验证损失未改善)')
                break
        
        total_time = time.time() - start_time
        print(f'训练完成! 总时间: {total_time/60:.1f}分钟')
        
        # 绘制损失曲线
        self.plot_losses()
    
    def plot_losses(self):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
        
        # 绘制损失
        ax1.plot(self.train_losses, label='训练损失', color='blue', linewidth=2)
        ax1.plot(self.val_losses, label='验证损失', color='red', linewidth=2)
        ax1.set_ylabel('损失')
        ax1.set_title('训练和验证损失')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 绘制学习率
        ax2.plot(self.learning_rates, label='学习率', color='green', linewidth=2)
        ax2.set_xlabel('轮次')
        ax2.set_ylabel('学习率')
        ax2.set_title('学习率变化')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_curves.png')
        plt.close()