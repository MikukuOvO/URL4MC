import torch
import math
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]
    
class Transformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, mask_ratio=0.75):
        super(Transformer, self).__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.mask_ratio = mask_ratio

        # 输入映射层
        self.input_projection = nn.Linear(1, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=input_dim)

        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 输出映射层
        self.output_projection = nn.Linear(d_model, 1)

    def forward(self, x):
        # x shape: (batch_size, input_dim)

        if self.training:
            mask = torch.bernoulli(torch.full(x.shape, 1 - self.mask_ratio)).to(device=x.device)
        else:
            # 如果是评估模式，不应用掩码
            mask = torch.ones_like(x).to(device=x.device)

        # 应用掩码
        masked_x = x * mask

        # 将输入reshape为(batch_size, input_dim, 1)并投影到d_model维度
        x = self.input_projection(masked_x.unsqueeze(-1))
        
        # Transformer期望的输入shape是(input_dim, batch_size, d_model)
        x = x.permute(1, 0, 2)
        x = self.positional_encoding(x)
        
        # 通过Transformer编码器
        encoded = self.transformer_encoder(x)
        
        # 将shape变回(batch_size, input_dim, d_model)
        encoded = encoded.permute(1, 0, 2)
        
        # 投影回原始输入维度
        output = self.output_projection(encoded).squeeze(-1)

        return output, mask
