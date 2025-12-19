"""
Transformer model for predicting bitcoin price movements
"""

import math
import torch
import torch.nn as nn

from config import D_MODEL, N_HEADS, N_LAYERS, D_FF, DROPOUT


class PositionalEncoding(nn.Module):
    """
    Adds position info to the sequence
    Without this, transformer doesn't know the order of data
    """
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        # create position encodings using sin/cos
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerModel(nn.Module):
    """
    Main model - predicts percentile of next hour return
    
    Input: sequence of features (batch, time, features)
    Output: percentile prediction 0-1 (batch, 1)
    """
    
    def __init__(self, input_dim):
        super().__init__()
        
        # project input features to model dimension
        self.input_proj = nn.Linear(input_dim, D_MODEL)
        self.input_norm = nn.LayerNorm(D_MODEL)
        
        # positional encoding
        self.pos_encoder = PositionalEncoding(D_MODEL)
        
        # transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=N_HEADS,
            dim_feedforward=D_FF,
            dropout=DROPOUT,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, N_LAYERS)
        
        # output layers
        self.output_norm = nn.LayerNorm(D_MODEL)
        self.dropout = nn.Dropout(DROPOUT)
        self.output_proj = nn.Linear(D_MODEL, 1)
        
        # initialize weights
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.normal_(self.output_proj.weight, mean=0, std=0.02)
        nn.init.constant_(self.output_proj.bias, 0.5)
    
    def forward(self, x):
        # project to model dimension
        x = self.input_proj(x)
        x = self.input_norm(x)
        
        # add position info
        x = self.pos_encoder(x)
        
        # pass through transformer
        x = self.transformer(x)
        
        # take last timestep only
        x = x[:, -1, :]
        
        # final layers
        x = self.output_norm(x)
        x = self.dropout(x)
        x = self.output_proj(x)
        
        # clamp to 0-1 range
        x = torch.clamp(x, 0, 1)
        
        return x


def load_model(path, input_dim, device):
    """Load saved model from file"""
    model = TransformerModel(input_dim)
    model = model.to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model