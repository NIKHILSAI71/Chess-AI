"""
Neural Network Architecture for Chess AI
=======================================

Implementation of AlphaZero-style policy-value network and NNUE evaluation.
This module provides the neural network components for the grandmaster-level chess engine.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from core.board import ChessBoard, Color, Piece, Square


@dataclass
class NetworkConfig:
    """Configuration for neural network architecture."""
    # ResNet backbone configuration
    num_blocks: int = 20  # Number of residual blocks
    filters: int = 256    # Number of filters per convolution
    
    # Input representation
    input_channels: int = 18  # 12 piece types + 6 auxiliary channels
    board_size: int = 8
    
    # Policy head
    policy_channels: int = 73  # 64 squares + 9 underpromotions per direction
    
    # Value head
    value_hidden: int = 256
    
    # Training parameters
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    batch_size: int = 32


class ConvBlock(nn.Module):
    """Convolutional block with batch normalization and ReLU."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                             padding=kernel_size//2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """Residual block for deep neural network."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class PolicyHead(nn.Module):
    """Policy head for move prediction."""
    
    def __init__(self, input_channels: int, policy_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, 32, 1, bias=False)
        self.bn = nn.BatchNorm2d(32)
        self.fc = nn.Linear(32 * 8 * 8, policy_channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn(self.conv(x)))
        x = x.view(x.size(0), -1)  # Flatten
        return self.fc(x)


class ValueHead(nn.Module):
    """Value head for position evaluation."""
    
    def __init__(self, input_channels: int, hidden_size: int):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, 32, 1, bias=False)
        self.bn = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(32 * 8 * 8, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn(self.conv(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        return torch.tanh(self.fc2(x))


class AlphaZeroNetwork(nn.Module):
    """
    AlphaZero-style policy-value network.
    
    Takes board position as input and outputs:
    - Policy: Probability distribution over legal moves
    - Value: Expected game outcome from current position
    """
    
    def __init__(self, config: NetworkConfig):
        super().__init__()
        self.config = config
        self.input_conv = ConvBlock(config.input_channels, config.filters)
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(config.filters) for _ in range(config.num_blocks)
        ])
        self.policy_head = PolicyHead(config.filters, config.policy_channels)
        self.value_head = ValueHead(config.filters, config.value_hidden)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor [batch_size, channels, 8, 8]
            
        Returns:
            policy: Policy logits [batch_size, policy_channels]
            value: Value prediction [batch_size, 1]
        """
        # Input processing
        x = self.input_conv(x)
        
        # Residual tower
        for block in self.residual_blocks:
            x = block(x)
        
        # Output heads
        policy = self.policy_head(x)
        value = self.value_head(x)
        
        return policy, value


class BoardEncoder:
    """
    Encodes chess board positions into neural network input format.
    
    Uses planes-based representation with separate channels for:
    - Each piece type and color (12 channels)
    - Castling rights (4 channels)
    - En passant square (1 channel)
    - Turn to move (1 channel)
    """
    
    def __init__(self):
        self.piece_to_channel = {
            (Color.WHITE, Piece.PAWN): 0,
            (Color.WHITE, Piece.KNIGHT): 1,
            (Color.WHITE, Piece.BISHOP): 2,
            (Color.WHITE, Piece.ROOK): 3,
            (Color.WHITE, Piece.QUEEN): 4,
            (Color.WHITE, Piece.KING): 5,
            (Color.BLACK, Piece.PAWN): 6,
            (Color.BLACK, Piece.KNIGHT): 7,
            (Color.BLACK, Piece.BISHOP): 8,
            (Color.BLACK, Piece.ROOK): 9,
            (Color.BLACK, Piece.QUEEN): 10,
            (Color.BLACK, Piece.KING): 11,
        }
    
    def encode_board(self, board: ChessBoard) -> np.ndarray:
        """
        Encode board position as multi-channel 8x8 representation.
        
        Args:
            board: Chess board to encode
            
        Returns:
            Encoded position as numpy array [18, 8, 8]
        """
        # Initialize input planes
        planes = np.zeros((18, 8, 8), dtype=np.float32)
        
        # Encode piece positions (12 channels)
        for (color, piece), channel in self.piece_to_channel.items():
            bitboard = board.bitboards[(color, piece)]
            for square in range(64):
                if bitboard.get_bit(square):
                    rank, file = square // 8, square % 8
                    planes[channel, rank, file] = 1.0
        
        # Encode castling rights (4 channels)
        if board.castling_rights.white_kingside:
            planes[12, :, :] = 1.0
        if board.castling_rights.white_queenside:
            planes[13, :, :] = 1.0
        if board.castling_rights.black_kingside:
            planes[14, :, :] = 1.0
        if board.castling_rights.black_queenside:
            planes[15, :, :] = 1.0
        
        # Encode en passant square (1 channel)
        if board.en_passant_square is not None:
            rank, file = board.en_passant_square // 8, board.en_passant_square % 8
            planes[16, rank, file] = 1.0
        
        # Encode turn to move (1 channel)
        if board.to_move == Color.WHITE:
            planes[17, :, :] = 1.0
        
        return planes
    
    def encode_batch(self, boards: List[ChessBoard]) -> torch.Tensor:
        """
        Encode a batch of board positions.
        
        Args:
            boards: List of chess boards to encode
            
        Returns:
            Batch tensor [batch_size, 18, 8, 8]
        """
        batch = np.stack([self.encode_board(board) for board in boards])
        return torch.from_numpy(batch)


class MoveEncoder:
    """
    Encodes chess moves for policy head training and inference.
    
    Maps legal moves to policy head output indices using move encoding:
    - Queen moves: 56 possible directions × 7 distances = 392 possibilities per square
    - Knight moves: 8 possible moves per square
    - Underpromotions: 3 pieces × 2 directions = 6 per square
    
    Total policy space is compressed to handle the most common moves efficiently.
    """
    
    def __init__(self):
        # Direction mappings for queen-like moves
        self.directions = [
            (-1, -1), (-1, 0), (-1, 1),  # Up-left, up, up-right
            (0, -1),           (0, 1),   # Left, right
            (1, -1),  (1, 0),  (1, 1)    # Down-left, down, down-right
        ]
        
        # Knight move offsets
        self.knight_moves = [
            (-2, -1), (-2, 1), (-1, -2), (-1, 2),
            (1, -2), (1, 2), (2, -1), (2, 1)
        ]
    
    def move_to_policy_index(self, move, board: ChessBoard) -> int:
        """
        Convert a move to policy head index.
        
        This is a simplified encoding - a full implementation would need
        to handle all possible moves more comprehensively.
        """
        from_square = move.from_square
        to_square = move.to_square
        
        # Simple encoding: just use the destination square for now
        # A full implementation would encode move types and directions
        return to_square
    
    def policy_to_moves(self, policy_probs: torch.Tensor, board: ChessBoard) -> List[Tuple[int, float]]:
        """
        Convert policy probabilities to list of (move_index, probability) pairs.
        
        Args:
            policy_probs: Softmax probabilities from policy head
            board: Current board position
            
        Returns:
            List of (move_index, probability) tuples
        """
        # This is a simplified implementation
        # A full version would properly decode the policy representation
        move_probs = []
        for i, prob in enumerate(policy_probs):
            if prob > 0.001:  # Only consider moves with reasonable probability
                move_probs.append((i, prob.item()))
        
        return sorted(move_probs, key=lambda x: x[1], reverse=True)


class NNUE(nn.Module):
    """
    NNUE (Efficiently Updatable Neural Network for Evaluation) implementation.
    
    Designed for fast incremental updates during alpha-beta search.
    Uses HalfKP features (King position + piece positions) for input.
    """
    
    def __init__(self, feature_size: int = 41024, hidden_size: int = 256):
        super().__init__()
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        
        # Feature transformer
        self.input_layer = nn.Linear(feature_size, hidden_size)
        
        # Neural network layers
        self.hidden1 = nn.Linear(hidden_size * 2, 32)  # * 2 for both sides
        self.hidden2 = nn.Linear(32, 32)
        self.output = nn.Linear(32, 1)
        
    def forward(self, white_features: torch.Tensor, black_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through NNUE.
        
        Args:
            white_features: Feature vector from white's perspective
            black_features: Feature vector from black's perspective
            
        Returns:
            Evaluation score
        """
        # Transform features
        white_hidden = F.relu(self.input_layer(white_features))
        black_hidden = F.relu(self.input_layer(black_features))
        
        # Concatenate both perspectives
        combined = torch.cat([white_hidden, black_hidden], dim=-1)
        
        # Process through network
        x = F.relu(self.hidden1(combined))
        x = F.relu(self.hidden2(x))
        
        return self.output(x)


class NeuralNetworkEvaluator:
    """
    High-level interface for neural network evaluation.
    
    Manages model loading, inference, and integration with the search engine.
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        self.device = torch.device(device)
        self.encoder = BoardEncoder()
        self.move_encoder = MoveEncoder()
        
        # Initialize with default configuration
        self.config = NetworkConfig()
        self.model = AlphaZeroNetwork(self.config).to(self.device)
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            print("Warning: No trained model loaded. Using random initialization.")
    
    def load_model(self, model_path: str):
        """Load trained model weights."""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            print(f"Loaded model from {model_path}")
        except Exception as e:
            print(f"Failed to load model: {e}")
    
    def save_model(self, model_path: str, epoch: int, loss: float):
        """Save model checkpoint."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'loss': loss,
            'config': self.config
        }, model_path)
    
    def evaluate_position(self, board: ChessBoard) -> float:
        """
        Evaluate a single position using the neural network.
        
        Args:
            board: Chess position to evaluate
            
        Returns:
            Evaluation score from the current player's perspective
        """
        self.model.eval()
        with torch.no_grad():
            # Encode position
            input_tensor = self.encoder.encode_batch([board]).to(self.device)
            
            # Get network prediction
            _, value = self.model(input_tensor)
            
            # Return evaluation from current player's perspective
            eval_score = value.item()
            if board.to_move == Color.BLACK:
                eval_score = -eval_score
            
            return eval_score
    
    def get_move_probabilities(self, board: ChessBoard) -> Dict[int, float]:
        """
        Get move probabilities for the current position.
        
        Args:
            board: Chess position to analyze
            
        Returns:
            Dictionary mapping move indices to probabilities
        """
        self.model.eval()
        with torch.no_grad():
            # Encode position
            input_tensor = self.encoder.encode_batch([board]).to(self.device)
            
            # Get network prediction
            policy_logits, _ = self.model(input_tensor)
            
            # Convert to probabilities
            policy_probs = F.softmax(policy_logits, dim=1)[0]
            
            # Convert to move probabilities
            move_probs = self.move_encoder.policy_to_moves(policy_probs, board)
            
            return dict(move_probs)
    
    def train_step(self, boards: List[ChessBoard], target_policies: List[np.ndarray], 
                   target_values: List[float], optimizer: torch.optim.Optimizer) -> float:
        """
        Perform one training step.
        
        Args:
            boards: Batch of board positions
            target_policies: Target policy distributions
            target_values: Target value labels
            optimizer: PyTorch optimizer
            
        Returns:
            Combined loss value
        """
        self.model.train()
        
        # Encode inputs
        input_tensor = self.encoder.encode_batch(boards).to(self.device)
        target_policies_tensor = torch.from_numpy(np.stack(target_policies)).to(self.device)
        target_values_tensor = torch.tensor(target_values, dtype=torch.float32).to(self.device)
        
        # Forward pass
        policy_logits, values = self.model(input_tensor)
        
        # Compute losses
        policy_loss = F.cross_entropy(policy_logits, target_policies_tensor)
        value_loss = F.mse_loss(values.squeeze(), target_values_tensor)
        
        # Combined loss
        total_loss = policy_loss + value_loss
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        return total_loss.item()


def create_default_network(device: str = "cpu") -> NeuralNetworkEvaluator:
    """
    Create a neural network evaluator with default configuration.
    
    Args:
        device: Device to run the network on ("cpu" or "cuda")
        
    Returns:
        Initialized neural network evaluator
    """
    return NeuralNetworkEvaluator(device=device)


# Example usage and testing
if __name__ == "__main__":
    # Test neural network components
    print("Testing Chess AI Neural Network...")
    
    # Create default network
    nn_eval = create_default_network()
    
    # Test with starting position
    from core.board import ChessBoard
    board = ChessBoard()
    
    # Test position evaluation
    eval_score = nn_eval.evaluate_position(board)
    print(f"Starting position evaluation: {eval_score:.3f}")
    
    # Test move probabilities
    move_probs = nn_eval.get_move_probabilities(board)
    print(f"Number of moves with probabilities: {len(move_probs)}")
    
    # Test board encoding
    encoder = BoardEncoder()
    encoded = encoder.encode_board(board)
    print(f"Encoded board shape: {encoded.shape}")
    print(f"Sum of piece planes: {encoded[:12].sum()}")  # Should be 32 (total pieces)
    
    print("Neural network tests completed successfully!")
