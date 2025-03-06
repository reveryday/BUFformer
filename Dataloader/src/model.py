import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # 注册为非参数缓冲区
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        return x + self.pe[:, :x.size(1), :]

class FeatureEncoder(nn.Module):
    """特征编码器，将物理特征转换为嵌入表示"""
    def __init__(self, energy_dim=1, atomic_dim=4, mfp_dim=4, engineered_dim=16, d_model=128, dropout=0.1):
        super(FeatureEncoder, self).__init__()
        
        # 特征嵌入
        self.energy_embedding = nn.Sequential(
            nn.Linear(energy_dim, d_model // 8),
            nn.GELU(),
            nn.LayerNorm(d_model // 8),
            nn.Dropout(dropout)
        )
        
        self.atomic_embedding = nn.Sequential(
            nn.Linear(atomic_dim, d_model // 4),
            nn.GELU(),
            nn.LayerNorm(d_model // 4),
            nn.Dropout(dropout)
        )
        
        self.mfp_embedding = nn.Sequential(
            nn.Linear(mfp_dim, d_model // 4),
            nn.GELU(),
            nn.LayerNorm(d_model // 4),
            nn.Dropout(dropout)
        )
        
        self.engineered_embedding = nn.Sequential(
            nn.Linear(engineered_dim, d_model // 2),
            nn.GELU(),
            nn.LayerNorm(d_model // 2),
            nn.Dropout(dropout)
        )
        
        # 特征交叉注意力
        self.feature_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # 特征融合
        self.feature_fusion = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, energy, atomic, mfp, engineered):
        # 嵌入特征
        energy_embed = self.energy_embedding(energy)
        atomic_embed = self.atomic_embedding(atomic)
        mfp_embed = self.mfp_embedding(mfp)
        engineered_embed = self.engineered_embedding(engineered)
        
        # 特征融合
        combined = torch.cat([energy_embed, atomic_embed, mfp_embed, engineered_embed], dim=1)
        
        # 添加序列维度并进行自注意力
        combined = combined.unsqueeze(1)  # [batch_size, 1, d_model]
        
        # 自注意力处理
        attn_output, _ = self.feature_attention(combined, combined, combined)
        
        # 特征融合
        output = self.feature_fusion(attn_output.squeeze(1))
        
        return output

class PhysicsConstraintLayer(nn.Module):
    """物理约束层，确保输出符合物理规律"""
    def __init__(self, d_model, dropout=0.1):
        super(PhysicsConstraintLayer, self).__init__()
        
        self.constraint_net = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.LayerNorm(d_model * 2),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model)
        )
    
    def forward(self, x):
        # 应用物理约束
        return self.constraint_net(x) + x  # 残差连接

class TransformerEncoderWithPhysics(nn.Module):
    """带有物理约束的Transformer编码器层"""
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super(TransformerEncoderWithPhysics, self).__init__()
        
        # 标准Transformer编码器层
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # 物理约束层
        self.physics_layer = PhysicsConstraintLayer(d_model, dropout)
        
        # 激活函数
        self.activation = nn.GELU()
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # 多头自注意力
        src2, _ = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # 前馈网络
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        # 物理约束
        src = self.physics_layer(src)
        
        return src

class TransformerPredictor(nn.Module):
    def __init__(self, input_dim=41, output_dim=6, d_model=256, nhead=8, 
                 num_encoder_layers=6, dim_feedforward=1024, dropout=0.1):
        super(TransformerPredictor, self).__init__()
        
        # 保存模型参数
        self.d_model = d_model
        self.input_dim = input_dim
        
        # 特征分离
        self.energy_dim = 1
        self.atomic_dim = 4
        self.mfp_dim = 4
        self.engineered_dim = input_dim - 1 - 4 - 4  # 工程特征维度
        
        # 特征编码器
        self.feature_encoder = FeatureEncoder(
            energy_dim=self.energy_dim,
            atomic_dim=self.atomic_dim,
            mfp_dim=self.mfp_dim,
            engineered_dim=self.engineered_dim,
            d_model=d_model,
            dropout=dropout
        )
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer编码器层
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderWithPhysics(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            ) for _ in range(num_encoder_layers)
        ])
        
        # 输出预测层
        self.output_layers = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.LayerNorm(d_model // 2),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim)
        )
        
        # 针对不同输出的专注层
        self.scatter_attention = nn.Linear(d_model, 3)
        self.no_scatter_attention = nn.Linear(d_model, 3)
        
        # 初始化参数
        self._init_parameters()
    
    def _init_parameters(self):
        """初始化模型参数"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src, tgt=None):
        """前向传播
        
        Args:
            src: 输入特征 [batch_size, input_dim]
            tgt: 目标特征 [batch_size, output_dim]，训练时使用，推理时为None
            
        Returns:
            输出预测 [batch_size, output_dim]
        """
        # 特征分离
        energy = src[:, 0:1]  # 能量
        atomic_numbers = src[:, [1,3,5,7]]  # 原子序数
        mfp_values = src[:, [2,4,6,8]]  # 自由程
        engineered_features = src[:, 9:]  # 工程特征
        
        # 特征编码
        x = self.feature_encoder(energy, atomic_numbers, mfp_values, engineered_features)
        
        # 扩展序列维度并添加位置编码
        # 将特征转换为序列形式 [batch_size, seq_len=1, d_model]
        x = x.unsqueeze(1)
        x = self.pos_encoder(x)
        
        # 通过Transformer编码器层
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
        
        # 去除序列维度 [batch_size, d_model]
        x = x.squeeze(1)
        
        # 输出预测
        # 分别为不同类型的累积因子生成注意力权重
        no_scatter_attn = F.softmax(self.no_scatter_attention(x), dim=1)
        scatter_attn = F.softmax(self.scatter_attention(x), dim=1)
        
        # 生成最终预测
        output = self.output_layers(x)
        
        # 分离不同类型的预测
        no_scatter_pred = output[:, :3]
        scatter_pred = output[:, 3:]
        
        # 应用注意力机制
        no_scatter_pred = no_scatter_pred * no_scatter_attn
        scatter_pred = scatter_pred * scatter_attn
        
        # 组合预测结果
        output = torch.cat([no_scatter_pred, scatter_pred], dim=1)
        
        return output