import torch
import torch.nn as nn
import math

class OneLayerTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=1, mlp_dim=512):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, d_model)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len)
        seq_len = x.size(1)
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = x + self.pos_encoding(seq_len)
        
        # Self-attention
        attn_output, _ = self.attention(x, x, x)
        x = self.layer_norm1(x + attn_output)
        
        # MLP
        mlp_output = self.mlp(x)
        x = self.layer_norm2(x + mlp_output)
        
        return x

    def get_mlp_activation(self, x):
        # Forward pass up to MLP input
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = x + self.pos_encoding(x.size(1))
        attn_output, _ = self.attention(x, x, x)
        x = self.layer_norm1(x + attn_output)
        
        # Return activation after first linear layer and ReLU
        return nn.ReLU()(self.mlp[0](x))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, seq_len):
        return self.pe[:seq_len]