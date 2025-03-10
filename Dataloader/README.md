# XSformer - 辐射屏蔽设计优化模型

## 项目简介

这是一个基于Transformer架构的辐射屏蔽设计优化模型，旨在通过能量、屏蔽材料和自由程等参数，准确预测六种累积因子（包括考虑散射和不考虑散射的注量累积因子、照射量累积因子和有效剂量累积因子），从而实现辐射屏蔽的优化设计。

## 模型特点

1. **物理特征分离处理**：针对能量、原子序数和自由程等物理特征进行单独的特征工程和嵌入
2. **特征交互层**：捕捉不同物理参数之间的复杂关系
3. **物理约束层**：确保模型预测结果符合辐射物理规律
4. **Transformer架构**：利用自注意力机制处理特征间的长距离依赖关系

## 物理概念

**自由程(Mean Free Path)**：是描述粒子在介质中平均行进距离的物理量，与能量和材料原子序数密切相关。自由程比直接使用厚度更能反映辐射与物质相互作用的物理本质，因此本模型直接使用自由程作为输入参数。

## 使用方法

### 环境要求

- Python 3.6+
- PyTorch 1.7+
- NumPy
- Pandas
- scikit-learn

### 训练模型

```bash
python main.py --mode train --data_path data.txt --batch_size 64 --epochs 200
```

### 评估模型

```bash
python main.py --mode evaluate --data_path data.txt --model_path best_model.pth
```

### 使用模型进行预测

```bash
python main.py --mode predict --data_path data.txt --model_path best_model.pth
```

## 数据格式

输入数据应为文本文件，每行包含以下字段（以空格分隔）：

1. 能量（MeV）
2. 第一层材料原子序数
3. 第一层材料厚度（自由程）
4. 第二层材料原子序数
5. 第二层材料厚度（自由程）
6. 第三层材料原子序数
7. 第三层材料厚度（自由程）
8. 第四层材料原子序数
9. 第四层材料厚度（自由程）
10-12. 不考虑散射的三种累积因子（注量、照射量、有效剂量）
13-15. 考虑散射的三种累积因子（注量、照射量、有效剂量）

## 项目简介

XSformer是一个基于Transformer架构的辐射屏蔽设计优化模型，旨在通过能量、屏蔽材料和厚度等参数，准确预测六种累积因子（包括考虑散射和不考虑散射的注量累积因子、照射量累积因子和有效剂量累积因子），从而实现辐射屏蔽的优化设计。

## 模型特点

1. **物理特征分离处理**：针对能量、原子序数和厚度等物理特征进行单独的特征工程和嵌入
2. **特征交互层**：捕捉不同物理参数之间的复杂关系
3. **物理约束层**：确保模型预测结果符合辐射物理规律
4. **Transformer架构**：利用自注意力机制处理特征间的长距离依赖关系
5. **物理约束损失函数**：在训练过程中引入物理约束，如能量与累积因子的负相关关系

## 优化亮点

相比原始模型，本次优化主要包括：

1. **数据预处理优化**：
   - 为不同物理特征设计单独的标准化器
   - 优化特征组合方式，更好地保留物理意义

2. **模型架构优化**：
   - 添加BatchNorm和Dropout层，提高模型泛化能力
   - 设计特征交互层，增强特征间的信息交流
   - 引入物理约束层，确保预测结果符合物理规律

3. **训练过程优化**：
   - 设计复合损失函数，同时考虑MSE和L1损失
   - 添加物理约束损失，引导模型学习符合物理规律的映射关系
   - 实现学习率调度器，动态调整学习率
   - 添加早停机制，避免过拟合
   - 实现梯度裁剪，防止梯度爆炸

4. **评估功能增强**：
   - 提供详细的评估指标，包括MSE、RMSE、MAE、R²和相对误差
   - 增加物理特性分析功能，探索能量与累积因子的关系
   - 优化可视化效果，直观展示预测结果和误差分布

## 使用方法

### 环境要求

- Python 3.6+
- PyTorch 1.7+
- NumPy
- Pandas
- Matplotlib
- scikit-learn

### 训练模型

```bash
python main.py --mode train --data_path data.txt --batch_size 64 --epochs 200 --learning_rate 0.001
```

### 评估模型

```bash
python main.py --mode evaluate --data_path data.txt --model_path best_model.pth
```

### 使用模型进行预测

```bash
python main.py --mode predict --data_path data.txt --model_path best_model.pth
```

### 主要参数说明

- `--mode`：运行模式，可选 train（训练）、evaluate（评估）、predict（预测）
- `--data_path`：数据文件路径
- `--model_path`：模型文件路径，用于评估或预测
- `--batch_size`：批次大小
- `--epochs`：训练轮数
- `--learning_rate`：学习率
- `--d_model`：模型维度
- `--nhead`：注意力头数
- `--num_encoder_layers`：编码器层数
- `--num_decoder_layers`：解码器层数
- `--dim_feedforward`：前馈网络维度
- `--dropout`：Dropout比例
- `--seed`：随机种子

## 数据格式

输入数据应为文本文件，每行包含以下字段（以空格分隔）：

1. 能量（MeV）
2. 第一层材料原子序数
3. 第一层材料厚度（自由程）
4. 第二层材料原子序数
5. 第二层材料厚度（自由程）
6. 第三层材料原子序数
7. 第三层材料厚度（自由程）
8. 第四层材料原子序数
9. 第四层材料厚度（自由程）
10-12. 不考虑散射的三种累积因子（注量、照射量、有效剂量）
13-15. 考虑散射的三种累积因子（注量、照射量、有效剂量）

## 物理约束

模型训练过程中引入了以下物理约束：

1. 能量越高，累积因子应该越小（负相关）
2. 不考虑散射的累积因子应小于考虑散射的累积因子
3. 所有累积因子均为正值

这些约束确保了模型预测结果符合辐射物理规律，提高了模型在实际辐射屏蔽设计中的可靠性。