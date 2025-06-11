#!/usr/bin/env python3
"""
Advanced Neural Network Architecture for Chess AI
================================================
Implements state-of-the-art neural network architectures including:
- Enhanced AlphaZero-style policy-value networks
- Vision Transformer (ViT) for chess positions
- NNUE (Efficiently Updatable Neural Networks)
- Hybrid architectures combining multiple approaches
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import math


@dataclass
class NetworkConfig:
    """Configuration for neural network architectures."""
    # Board representation
    board_size: int = 8
    input_channels: int = 119  # Enhanced feature planes
    
    # CNN/ResNet configuration
    cnn_filters: int = 256
    cnn_blocks: int = 20
    
    # Transformer configuration  
    transformer_dim: int = 512
    transformer_heads: int = 8
    transformer_layers: int = 6
    
    # NNUE configuration
    nnue_input_size: int = 768  # HalfKP features
    nnue_hidden_size: int = 256
    
    # Output configuration
    policy_size: int = 4096  # All possible moves
    value_size: int = 1
    
    # Training configuration
    dropout_rate: float = 0.1
    batch_norm: bool = True
    activation: str = 'relu'


class ResidualBlock(nn.Module):
    """Enhanced residual block with optional squeeze-and-excitation."""
    
    def __init__(self, channels: int, use_se: bool = True, reduction: int = 16):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
        # Squeeze-and-Excitation module
        self.use_se = use_se
        if use_se:
            self.se = SqueezeExcitation(channels, reduction)
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        if self.use_se:
            out = self.se(out)
        
        out += residual
        return F.relu(out)


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation module for channel attention."""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.fc1 = nn.Conv2d(channels, channels // reduction, 1)
        self.fc2 = nn.Conv2d(channels // reduction, channels, 1)
    
    def forward(self, x):
        scale = F.adaptive_avg_pool2d(x, 1)
        scale = F.relu(self.fc1(scale))
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer architecture."""
    
    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class ChessTransformer(nn.Module):
    """Vision Transformer adapted for chess positions."""
    
    def __init__(self, config: NetworkConfig):
        super().__init__()
        self.config = config
        self.patch_size = 1  # Each square is a patch
        self.num_patches = 64
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            config.input_channels, 
            config.transformer_dim, 
            kernel_size=self.patch_size,
            stride=self.patch_size
        )
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(config.transformer_dim, self.num_patches + 1)
        
        # Class token for global representation
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.transformer_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.transformer_dim,
            nhead=config.transformer_heads,
            dim_feedforward=config.transformer_dim * 4,
            dropout=config.dropout_rate,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, config.transformer_layers)
        
        # Output layers
        self.policy_head = nn.Linear(config.transformer_dim, config.policy_size)
        self.value_head = nn.Linear(config.transformer_dim, 1)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Patch embedding
        x = self.patch_embed(x)  # [B, dim, 8, 8]
        x = x.flatten(2).transpose(1, 2)  # [B, 64, dim]
        
        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [B, 65, dim]
        
        # Positional encoding
        x = self.pos_encoding(x)
        
        # Transformer encoding
        x = self.transformer(x)
        
        # Use class token for global features
        global_features = x[:, 0]
        
        # Output heads
        policy = self.policy_head(global_features)
        value = torch.tanh(self.value_head(global_features))
        
        return policy, value


class NNUE(nn.Module):
    """Efficiently Updatable Neural Network for fast evaluation."""
    
    def __init__(self, config: NetworkConfig):
        super().__init__()
        self.config = config
        
        # Feature transformer
        self.feature_transformer = nn.Sequential(
            nn.Linear(config.nnue_input_size, config.nnue_hidden_size),
            nn.ReLU(),
            nn.Linear(config.nnue_hidden_size, config.nnue_hidden_size),
            nn.ReLU()
        )
        
        # Accumulator network
        self.accumulator = nn.Sequential(
            nn.Linear(config.nnue_hidden_size * 2, config.nnue_hidden_size),
            nn.ReLU(),
            nn.Linear(config.nnue_hidden_size, config.nnue_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(config.nnue_hidden_size // 2, 1)
        )
    
    def forward(self, white_features, black_features):
        """
        Forward pass with separate feature sets for white and black perspectives.
        """
        white_transformed = self.feature_transformer(white_features)
        black_transformed = self.feature_transformer(black_features)
        
        # Combine perspectives
        combined = torch.cat([white_transformed, black_transformed], dim=-1)
        
        # Final evaluation
        evaluation = self.accumulator(combined)
        return torch.tanh(evaluation)


class HybridChessNetwork(nn.Module):
    """
    Hybrid architecture combining CNN backbone with transformer attention
    and optional NNUE evaluation for maximum performance.
    """
    
    def __init__(self, config: NetworkConfig, use_nnue: bool = False):
        super().__init__()
        self.config = config
        self.use_nnue = use_nnue
        
        # CNN backbone for spatial features
        self.input_conv = nn.Conv2d(config.input_channels, config.cnn_filters, 3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(config.cnn_filters)
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(config.cnn_filters, use_se=True) 
            for _ in range(config.cnn_blocks)
        ])
        
        # Transformer for global attention
        self.transformer_proj = nn.Conv2d(config.cnn_filters, config.transformer_dim, 1)
        self.transformer = ChessTransformer(config)
        
        # Policy head with spatial awareness
        self.policy_conv = nn.Conv2d(config.cnn_filters, 128, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(128)
        self.policy_fc = nn.Linear(128 * 64, config.policy_size)
        
        # Value head
        self.value_conv = nn.Conv2d(config.cnn_filters, 32, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * 64, 256)
        self.value_fc2 = nn.Linear(256, 1)
        
        # Optional NNUE for fast evaluation
        if use_nnue:
            self.nnue = NNUE(config)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(config.dropout_rate)
    
    def forward(self, x, nnue_features=None):
        batch_size = x.size(0)
        
        # CNN backbone
        x = F.relu(self.input_bn(self.input_conv(x)))
        
        # Residual blocks with skip connections
        for block in self.res_blocks:
            x = block(x)
        
        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(batch_size, -1)
        policy = self.dropout(policy)
        policy = self.policy_fc(policy)
        
        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(batch_size, -1)
        value = self.dropout(value)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        # Optional NNUE evaluation
        if self.use_nnue and nnue_features is not None:
            white_features, black_features = nnue_features
            nnue_value = self.nnue(white_features, black_features)
            # Combine CNN and NNUE evaluations
            value = 0.7 * value + 0.3 * nnue_value
        
        return policy, value


class EnhancedAlphaZeroNetwork(nn.Module):
    """
    Enhanced AlphaZero network with advanced features and optimizations.
    """
    
    def __init__(self, config: NetworkConfig):
        super().__init__()
        self.config = config
        
        # Feature extraction with multiple scales
        self.multi_scale_conv = nn.ModuleList([
            nn.Conv2d(config.input_channels, config.cnn_filters // 4, kernel_size=k, padding=k//2)
            for k in [1, 3, 5, 7]
        ])
        
        # Main CNN backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(config.cnn_filters, config.cnn_filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(config.cnn_filters),
            nn.ReLU()
        )
        
        # Residual tower
        self.res_tower = nn.ModuleList([
            ResidualBlock(config.cnn_filters, use_se=True)
            for _ in range(config.cnn_blocks)
        ])
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=config.cnn_filters,
            num_heads=8,
            batch_first=True
        )
        
        # Policy head with move type classification
        self.policy_head = nn.Sequential(
            nn.Conv2d(config.cnn_filters, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * 64, config.policy_size)
        )
        
        # Value head with auxiliary outputs
        self.value_head = nn.Sequential(
            nn.Conv2d(config.cnn_filters, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 64, 256),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(256, 1),
            nn.Tanh()
        )
        
        # Auxiliary heads for richer training signal
        self.material_head = nn.Linear(256, 1)  # Material balance
        self.phase_head = nn.Linear(256, 3)     # Game phase (opening/middle/end)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Multi-scale feature extraction
        multi_scale_features = []
        for conv in self.multi_scale_conv:
            multi_scale_features.append(conv(x))
        x = torch.cat(multi_scale_features, dim=1)
        
        # Main backbone
        x = self.backbone(x)
        
        # Residual tower
        for block in self.res_tower:
            x = block(x)
        
        # Global attention
        x_flat = x.view(batch_size, x.size(1), -1).transpose(1, 2)
        attended, _ = self.attention(x_flat, x_flat, x_flat)
        x_attended = attended.transpose(1, 2).view_as(x)
        x = x + 0.1 * x_attended  # Residual connection
        
        # Policy output
        policy = self.policy_head(x)
        
        # Value and auxiliary outputs
        value_features = x.view(batch_size, x.size(1), -1).mean(dim=2)
        value = self.value_head(x)
        
        # Auxiliary predictions
        material = self.material_head(value_features)
        phase = F.softmax(self.phase_head(value_features), dim=1)
        
        return {
            'policy': policy,
            'value': value,
            'material': material,
            'phase': phase
        }


class ChessFeatureExtractor:
    """
    Advanced feature extraction for chess positions.
    Creates rich input representations for neural networks.
    """
    
    def __init__(self):
        self.piece_to_plane = {
            'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,  # White pieces
            'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11  # Black pieces
        }
    
    def extract_features(self, board, history_length: int = 8) -> torch.Tensor:
        """
        Extract comprehensive features from chess position.
        
        Features include:
        - Piece positions (12 planes)
        - Repetition count (3 planes)
        - Color to move (1 plane)
        - Castling rights (4 planes)
        - En passant (1 plane)
        - Move count (1 plane)
        - Piece attacks (24 planes)
        - Historical positions (8 * 12 = 96 planes)
        Total: 119 planes
        """
        features = torch.zeros(119, 8, 8)
        
        # Current position piece planes
        for row in range(8):
            for col in range(8):
                piece = board.get_piece_at(row, col)
                if piece and piece in self.piece_to_plane:
                    features[self.piece_to_plane[piece], row, col] = 1
        
        # Color to move
        if board.current_player == 'white':
            features[12, :, :] = 1
        
        # Castling rights
        castling_rights = board.get_castling_rights()
        if castling_rights['white_kingside']:
            features[13, :, :] = 1
        if castling_rights['white_queenside']:
            features[14, :, :] = 1
        if castling_rights['black_kingside']:
            features[15, :, :] = 1
        if castling_rights['black_queenside']:
            features[16, :, :] = 1
        
        # En passant
        en_passant = board.get_en_passant_square()
        if en_passant:
            row, col = en_passant
            features[17, row, col] = 1
        
        # Move count (normalized)
        move_count = min(board.move_count / 100.0, 1.0)
        features[18, :, :] = move_count
        
        # Repetition count
        rep_count = board.get_repetition_count()
        if rep_count >= 1:
            features[19, :, :] = 1
        if rep_count >= 2:
            features[20, :, :] = 1
        if rep_count >= 3:
            features[21, :, :] = 1
        
        # Piece attacks
        white_attacks, black_attacks = self.get_attack_maps(board)
        features[22:34] = white_attacks
        features[34:46] = black_attacks
        
        # Historical positions (last 8 positions)
        history = board.get_position_history(history_length)
        for i, hist_board in enumerate(history):
            if hist_board:
                for row in range(8):
                    for col in range(8):
                        piece = hist_board.get_piece_at(row, col)
                        if piece and piece in self.piece_to_plane:
                            plane_idx = 46 + i * 12 + self.piece_to_plane[piece]
                            features[plane_idx, row, col] = 1
        
        return features.unsqueeze(0)  # Add batch dimension
    
    def get_attack_maps(self, board) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate attack maps for both colors."""
        white_attacks = torch.zeros(12, 8, 8)
        black_attacks = torch.zeros(12, 8, 8)
        
        # This would need to be implemented based on your board representation
        # For now, return zeros
        return white_attacks, black_attacks
    
    def extract_nnue_features(self, board) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract HalfKP features for NNUE."""
        # Simplified NNUE feature extraction
        # In practice, this would use the HalfKP (Half of King-Piece) feature set
        white_features = torch.zeros(768)
        black_features = torch.zeros(768)
        
        # This would need proper implementation based on piece positions
        # relative to both kings
        
        return white_features.unsqueeze(0), black_features.unsqueeze(0)


def create_network(architecture: str = "hybrid", config: Optional[NetworkConfig] = None) -> nn.Module:
    """
    Factory function to create different network architectures.
    
    Args:
        architecture: Type of network ('hybrid', 'alphazero', 'transformer', 'nnue')
        config: Network configuration
    
    Returns:
        Configured neural network
    """
    if config is None:
        config = NetworkConfig()
    
    if architecture == "hybrid":
        return HybridChessNetwork(config, use_nnue=True)
    elif architecture == "alphazero":
        return EnhancedAlphaZeroNetwork(config)
    elif architecture == "transformer":
        return ChessTransformer(config)
    elif architecture == "nnue":
        return NNUE(config)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


# Export main classes
__all__ = [
    'NetworkConfig',
    'HybridChessNetwork', 
    'EnhancedAlphaZeroNetwork',
    'ChessTransformer',
    'NNUE',
    'ChessFeatureExtractor',
    'create_network'
]
