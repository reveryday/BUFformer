import torch
import torch.nn as nn
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

class TransformerPredictor(nn.Module):
    def __init__(self, input_dim=9, output_dim=6, d_model=64, nhead=8, 
                 num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=256, dropout=0.1):
        super(TransformerPredictor, self).__init__()
        
        # 物理特征处理层
        self.energy_embedding = nn.Sequential(
            nn.Linear(1, d_model // 4),
            nn.ReLU(),
            nn.BatchNorm1d(d_model // 4),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, d_model // 4)
        )
        
        self.material_embedding = nn.Sequential(
            nn.Linear(4, d_model // 4),
            nn.ReLU(),
            nn.BatchNorm1d(d_model // 4),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, d_model // 4)
        )
        
        self.mfp_embedding = nn.Sequential(
            nn.Linear(4, d_model // 4),
            nn.ReLU(),
            nn.BatchNorm1d(d_model // 4),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, d_model // 4)
        )
        
        # 特征交互层
        self.interaction_layer = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.BatchNorm1d(d_model * 2),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )
        
        # 物理约束层
        self.physics_constraint = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.BatchNorm1d(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )
        
        # 特征融合层
        self.feature_fusion = nn.Linear(d_model, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer编码器和解码器
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                      dim_feedforward=dim_feedforward, dropout=dropout,
                                      batch_first=True),
            num_layers=num_encoder_layers
        )
        
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, 
                                      dim_feedforward=dim_feedforward, dropout=dropout,
                                      batch_first=True),
            num_layers=num_decoder_layers
        )
        
        # 输出层
        self.output_projection = nn.Linear(d_model, output_dim)
        
        # 初始化参数
        self._init_parameters()
        
        # 模型维度
        self.d_model = d_model
        
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
        # 分离物理特征
        energy = src[:, 0:1]  # 能量
        atomic_numbers = src[:, [1,3,5,7]]  # 原子序数
        mfp_values = src[:, [2,4,6,8]]  # 自由程
        
        # 特征嵌入
        energy_features = self.energy_embedding(energy)
        material_features = self.material_embedding(atomic_numbers)
        thickness_features = self.mfp_embedding(mfp_values)
        
        # 特征融合
        combined_features = torch.cat([energy_features, material_features, thickness_features], dim=1)
        src = self.feature_fusion(combined_features)
        
        # 添加特征交互
        src = self.interaction_layer(src)
        
        # 应用物理约束
        src = self.physics_constraint(src)
        
        # 缩放并添加位置编码
        src = src.unsqueeze(1) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        
        # 通过编码器 (batch_first=True，不需要调整维度顺序)
        memory = self.transformer_encoder(src)
        
        # 创建目标序列或使用传入的目标
        if tgt is None:
            # 推理模式：使用全零向量作为初始目标
            tgt = torch.zeros(src.size(0), src.size(1), self.d_model, device=src.device)
        else:
            # 训练模式：使用真实目标
            tgt = tgt.unsqueeze(1)  # [batch_size, 1, output_dim]
            tgt = nn.Linear(tgt.size(-1), self.d_model).to(src.device)(tgt) * math.sqrt(self.d_model)
            tgt = self.pos_encoder(tgt)
        
        # 通过解码器 (batch_first=True，不需要调整维度顺序)
        output = self.transformer_decoder(tgt, memory)
        
        # 投影到输出维度 [batch_size, 1, output_dim]
        output = self.output_projection(output)
        
        # 去除序列维度 [batch_size, output_dim]
        output = output.squeeze(1)
        
        return output