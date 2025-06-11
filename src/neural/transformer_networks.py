#!/usr/bin/env python3
"""
Enhanced Neural Network Architectures for Chess AI
===============================================
Cutting-edge architectures including:
- Vision Transformer for Chess (ViTChess)
- Hybrid CNN-Transformer networks
- NNUE (Efficiently Updatable Neural Networks)
- Attention-based position encoding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List
import math
from dataclasses import dataclass

@dataclass
class NetworkConfig:
    """Configuration for neural network architectures"""
    # Input dimensions
    board_size: int = 8
    num_piece_types: int = 12  # 6 piece types * 2 colors
    num_channels: int = 18     # Piece channels + auxiliary info
    
    # Vision Transformer config
    patch_size: int = 2        # 2x2 patches for 8x8 board
    embed_dim: int = 256
    num_heads: int = 8
    num_transformer_layers: int = 6
    mlp_ratio: int = 4
    
    # CNN config
    cnn_channels: List[int] = None
    cnn_kernel_sizes: List[int] = None
    
    # Policy head config
    policy_channels: int = 256
    num_policy_outputs: int = 4096  # All possible moves (64*64)
    
    # Value head config
    value_hidden: int = 256
    
    # Training config
    dropout: float = 0.1
    
    def __post_init__(self):
        if self.cnn_channels is None:
            self.cnn_channels = [32, 64, 128, 256]
        if self.cnn_kernel_sizes is None:
            self.cnn_kernel_sizes = [3, 3, 3, 3]


class PositionalEncoding(nn.Module):
    """2D positional encoding for chess board"""
    
    def __init__(self, embed_dim: int, board_size: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.board_size = board_size
        
        # Create 2D positional encodings
        pe = torch.zeros(board_size, board_size, embed_dim)
        
        # Row encodings
        for pos in range(board_size):
            for i in range(0, embed_dim, 4):
                pe[pos, :, i] = math.sin(pos / (10000 ** (i / embed_dim)))
                pe[pos, :, i + 1] = math.cos(pos / (10000 ** (i / embed_dim)))
        
        # Column encodings
        for pos in range(board_size):
            for i in range(2, embed_dim, 4):
                pe[:, pos, i] = math.sin(pos / (10000 ** (i / embed_dim)))
                pe[:, pos, i + 1] = math.cos(pos / (10000 ** (i / embed_dim)))
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """Add positional encoding to input"""
        batch_size = x.size(0)
        return x + self.pe.unsqueeze(0).expand(batch_size, -1, -1, -1)


class ChessViTBlock(nn.Module):
    """Vision Transformer block adapted for chess"""
    
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden = embed_dim * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # Self-attention with residual connection
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        
        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))
        return x


class ViTChess(nn.Module):
    """Vision Transformer adapted for chess position analysis"""
    
    def __init__(self, config: NetworkConfig):
        super().__init__()
        self.config = config
        self.patch_size = config.patch_size
        self.embed_dim = config.embed_dim
        
        # Patch embedding
        patch_dim = config.num_channels * (config.patch_size ** 2)
        self.patch_embed = nn.Linear(patch_dim, config.embed_dim)
        
        # Number of patches
        self.num_patches = (config.board_size // config.patch_size) ** 2
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(config.embed_dim, config.board_size // config.patch_size)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            ChessViTBlock(config.embed_dim, config.num_heads, config.mlp_ratio, config.dropout)
            for _ in range(config.num_transformer_layers)
        ])
        
        self.norm = nn.LayerNorm(config.embed_dim)
        
        # Global attention pooling
        self.global_pool = nn.MultiheadAttention(config.embed_dim, config.num_heads, batch_first=True)
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.embed_dim))
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(config.embed_dim, config.policy_channels),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.policy_channels, config.num_policy_outputs)
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(config.embed_dim, config.value_hidden),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.value_hidden, 1),
            nn.Tanh()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier/Kaiming initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def patchify(self, x):
        """Convert input to patches"""
        batch_size, channels, height, width = x.shape
        assert height == width == self.config.board_size
        
        # Reshape to patches
        patch_h = patch_w = self.patch_size
        patches = x.unfold(2, patch_h, patch_h).unfold(3, patch_w, patch_w)
        patches = patches.contiguous().view(batch_size, channels, self.num_patches, patch_h * patch_w)
        patches = patches.permute(0, 2, 1, 3).contiguous()
        patches = patches.view(batch_size, self.num_patches, -1)
        
        return patches
    
    def forward(self, x):
        """Forward pass through ViT"""
        batch_size = x.size(0)
        
        # Convert to patches and embed
        patches = self.patchify(x)  # [batch, num_patches, patch_dim]
        x = self.patch_embed(patches)  # [batch, num_patches, embed_dim]
        
        # Add positional encoding
        grid_size = self.config.board_size // self.patch_size
        x = x.view(batch_size, grid_size, grid_size, self.embed_dim)
        x = self.pos_encoding(x)
        x = x.view(batch_size, self.num_patches, self.embed_dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # Global attention pooling
        cls_token = x[:, 0:1]  # [batch, 1, embed_dim]
        patch_tokens = x[:, 1:]  # [batch, num_patches, embed_dim]
        
        pooled, _ = self.global_pool(cls_token, patch_tokens, patch_tokens)
        pooled = pooled.squeeze(1)  # [batch, embed_dim]
        
        # Policy and value heads
        policy = self.policy_head(pooled)
        value = self.value_head(pooled)
        
        return policy, value.squeeze(-1)


class NNUE(nn.Module):
    """Efficiently Updatable Neural Network for Chess Evaluation"""
    
    def __init__(self, input_features: int = 768, hidden_size: int = 256):
        super().__init__()
        self.input_features = input_features
        self.hidden_size = hidden_size
        
        # Feature transformer
        self.feature_transformer = nn.Sequential(
            nn.Linear(input_features, hidden_size),
            nn.ClippedReLU()  # Custom activation
        )
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),  # *2 for both sides
            nn.ClippedReLU(),
            nn.Linear(32, 32),
            nn.ClippedReLU(),
            nn.Linear(32, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize with small weights for stability"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.uniform_(module.weight, -0.1, 0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, white_features, black_features):
        """Forward pass with separate feature sets for each side"""
        white_transformed = self.feature_transformer(white_features)
        black_transformed = self.feature_transformer(black_features)
        
        # Concatenate transformed features
        combined = torch.cat([white_transformed, black_transformed], dim=-1)
        
        # Output evaluation
        evaluation = self.output_layers(combined)
        return evaluation.squeeze(-1)


class ClippedReLU(nn.Module):
    """ReLU activation clipped at maximum value for NNUE"""
    
    def __init__(self, max_val: float = 127.0):
        super().__init__()
        self.max_val = max_val
    
    def forward(self, x):
        return torch.clamp(F.relu(x), max=self.max_val)


class HybridChessNetwork(nn.Module):
    """Hybrid CNN-Transformer network for chess"""
    
    def __init__(self, config: NetworkConfig):
        super().__init__()
        self.config = config
        
        # CNN backbone for local feature extraction
        self.cnn_layers = nn.ModuleList()
        in_channels = config.num_channels
        
        for out_channels, kernel_size in zip(config.cnn_channels, config.cnn_kernel_sizes):
            self.cnn_layers.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Dropout2d(config.dropout)
            ))
            in_channels = out_channels
        
        # Transition to transformer
        self.cnn_to_transformer = nn.Conv2d(in_channels, config.embed_dim, 1)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(config.embed_dim, config.board_size)
        
        # Transformer layers
        self.transformer_blocks = nn.ModuleList([
            ChessViTBlock(config.embed_dim, config.num_heads, config.mlp_ratio, config.dropout)
            for _ in range(config.num_transformer_layers)
        ])
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Heads
        self.policy_head = nn.Sequential(
            nn.Linear(config.embed_dim, config.policy_channels),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.policy_channels, config.num_policy_outputs)
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(config.embed_dim, config.value_hidden),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.value_hidden, 1),
            nn.Tanh()
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # CNN feature extraction
        for layer in self.cnn_layers:
            x = layer(x)
        
        # Transition to transformer dimension
        x = self.cnn_to_transformer(x)  # [batch, embed_dim, 8, 8]
        
        # Prepare for transformer
        x = x.permute(0, 2, 3, 1)  # [batch, 8, 8, embed_dim]
        x = self.pos_encoding(x)
        
        # Flatten spatial dimensions for transformer
        x = x.view(batch_size, 64, self.config.embed_dim)  # [batch, 64, embed_dim]
        
        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Global pooling
        x = x.mean(dim=1)  # [batch, embed_dim]
        
        # Policy and value heads
        policy = self.policy_head(x)
        value = self.value_head(x)
        
        return policy, value.squeeze(-1)


class ResidualBlock(nn.Module):
    """Residual block for CNN backbone"""
    
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


def create_advanced_network(architecture: str = "hybrid", config: Optional[NetworkConfig] = None) -> nn.Module:
    """
    Factory function to create advanced neural network architectures
    
    Args:
        architecture: Type of architecture ('vit', 'hybrid', 'nnue')
        config: Network configuration
        
    Returns:
        Neural network model
    """
    if config is None:
        config = NetworkConfig()
    
    if architecture == "vit":
        return ViTChess(config)
    elif architecture == "hybrid":
        return HybridChessNetwork(config)
    elif architecture == "nnue":
        return NNUE()
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


# Neural network wrapper for MCTS integration
class NeuralNetworkWrapper:
    """Wrapper class for neural network integration with MCTS"""
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    def predict(self, board_state) -> Tuple[np.ndarray, float]:
        """
        Predict policy and value for board state
        
        Args:
            board_state: Chess board state
            
        Returns:
            Tuple of (policy_probabilities, value_estimate)
        """
        try:
            # Convert board to tensor
            input_tensor = self.board_to_tensor(board_state)
            input_tensor = input_tensor.to(self.device)
            
            with torch.no_grad():
                policy_logits, value = self.model(input_tensor)
                
                # Convert to probabilities
                policy_probs = F.softmax(policy_logits, dim=-1)
                
                return policy_probs.cpu().numpy()[0], value.cpu().item()
        
        except Exception as e:
            print(f"Neural network prediction error: {e}")
            # Return uniform policy and neutral value
            return np.ones(4096) / 4096, 0.0
    
    def board_to_tensor(self, board_state) -> torch.Tensor:
        """Convert board state to neural network input tensor"""
        # This should be implemented based on your board representation
        # For now, return a dummy tensor
        batch_size = 1
        channels = 18
        height = width = 8
        
        # Create dummy input - replace with actual board encoding
        tensor = torch.zeros(batch_size, channels, height, width)
        
        return tensor
