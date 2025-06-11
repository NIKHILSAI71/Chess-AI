"""
Self-Play Training Infrastructure
================================

Implementation of AlphaZero-style self-play training for the chess neural network.
Generates training data through self-play games and trains the network iteratively.
"""

import os
import json
import time
import random
import threading
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.board import ChessBoard, Move, Color
from core.moves import MoveGenerator
from neural.network import NeuralNetworkEvaluator, NetworkConfig, AlphaZeroNetwork, BoardEncoder
from neural.mcts import MCTS, MCTSConfig, MCTSSearchEngine


@dataclass
class TrainingConfig:
    """Configuration for self-play training."""
    # Training parameters
    num_iterations: int = 100           # Number of training iterations
    games_per_iteration: int = 25       # Self-play games per iteration
    training_epochs: int = 10           # Training epochs per iteration
    batch_size: int = 32                # Batch size for training
    
    # Self-play parameters
    mcts_simulations: int = 400         # MCTS simulations per move
    temperature_moves: int = 30         # Moves to use temperature > 0
    temperature: float = 1.0            # Temperature for move selection
    
    # Model parameters
    learning_rate: float = 0.001        # Learning rate
    weight_decay: float = 1e-4          # L2 regularization
    lr_decay_steps: int = 100000        # Steps for learning rate decay
    lr_decay_rate: float = 0.1          # Learning rate decay factor
    
    # Data management
    max_training_samples: int = 500000  # Maximum training samples to keep
    validation_split: float = 0.1       # Fraction for validation
    
    # Checkpointing
    save_frequency: int = 5             # Save model every N iterations
    eval_frequency: int = 10            # Evaluate model every N iterations
    
    # Paths
    model_dir: str = "models"
    data_dir: str = "training_data"
    log_dir: str = "logs"


@dataclass
class GameResult:
    """Result of a self-play game."""
    positions: List[np.ndarray]         # Board positions (encoded)
    move_probabilities: List[np.ndarray] # MCTS move probabilities
    values: List[float]                 # Game outcome from each position
    game_length: int                    # Number of moves in the game
    winner: Optional[Color]             # Winner of the game (None for draw)


class TrainingDataset(Dataset):
    """
    PyTorch dataset for chess training data.
    
    Manages positions, move probabilities, and values from self-play games.
    """
    
    def __init__(self, positions: List[np.ndarray], policies: List[np.ndarray], values: List[float]):
        self.positions = positions
        self.policies = policies
        self.values = values
        
        assert len(positions) == len(policies) == len(values), "Data lengths must match"
    
    def __len__(self) -> int:
        return len(self.positions)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        position = torch.from_numpy(self.positions[idx]).float()
        policy = torch.from_numpy(self.policies[idx]).float()
        value = torch.tensor(self.values[idx]).float()
        
        return position, policy, value
    
    def add_game_data(self, game_result: GameResult):
        """Add data from a completed game."""
        self.positions.extend(game_result.positions)
        self.policies.extend(game_result.move_probabilities)
        self.values.extend(game_result.values)
    
    def trim_to_size(self, max_size: int):
        """Keep only the most recent training samples."""
        if len(self.positions) > max_size:
            # Keep the most recent samples
            self.positions = self.positions[-max_size:]
            self.policies = self.policies[-max_size:]
            self.values = self.values[-max_size:]


class SelfPlayEngine:
    """
    Engine for generating self-play training games.
    
    Uses MCTS with the current neural network to play games against itself.
    """
    
    def __init__(self, neural_net: NeuralNetworkEvaluator, move_generator: MoveGenerator,
                 mcts_config: MCTSConfig):
        self.neural_net = neural_net
        self.move_generator = move_generator
        self.mcts_config = mcts_config
        self.encoder = BoardEncoder()
    
    def play_game(self, temperature_moves: int = 30, temperature: float = 1.0) -> GameResult:
        """
        Play a complete self-play game.
        
        Args:
            temperature_moves: Number of moves to use temperature > 0
            temperature: Temperature for move selection
            
        Returns:
            GameResult containing training data from the game
        """
        board = ChessBoard()
        positions = []
        move_probs = []
        
        move_count = 0
        
        while True:
            # Check for terminal position
            legal_moves = self.move_generator.generate_legal_moves(board)
            
            if len(legal_moves) == 0:
                # Game over - checkmate or stalemate
                if self.move_generator.is_in_check(board, board.to_move):
                    # Checkmate
                    winner = Color.BLACK if board.to_move == Color.WHITE else Color.WHITE
                else:
                    # Stalemate
                    winner = None
                break
            
            # Check for draw conditions
            if (board.halfmove_clock >= 100 or          # 50-move rule
                board.is_threefold_repetition() or     # Repetition
                move_count >= 300):                     # Max game length
                winner = None
                break
            
            # Encode current position
            position = self.encoder.encode_board(board)
            positions.append(position)
            
            # Perform MCTS search
            mcts = MCTS(self.neural_net, self.move_generator, self.mcts_config)
            
            # Use temperature for early moves, then greedy selection
            current_temp = temperature if move_count < temperature_moves else 0.0
            
            # Get move probabilities from MCTS
            best_move, _ = mcts.search(board)
            
            # Get visit count distribution as training target
            visit_counts = mcts.root.get_child_visit_counts()
            total_visits = sum(visit_counts.values())
            
            # Create policy target (simplified - maps to move indices)
            policy_target = np.zeros(64)  # Simplified: just destination squares
            for move_str, visits in visit_counts.items():
                # Parse move to get destination square (simplified)
                try:
                    move = self._parse_move_string(move_str)
                    if move:
                        policy_target[move.to_square] = visits / total_visits
                except:
                    pass
            
            move_probs.append(policy_target)
            
            # Select move based on temperature
            if current_temp > 0:
                # Probabilistic selection
                moves = []
                probs = []
                for move_str, visits in visit_counts.items():
                    move = self._parse_move_string(move_str)
                    if move:
                        moves.append(move)
                        probs.append(visits)
                
                if probs:
                    probs = np.array(probs, dtype=np.float64)
                    if current_temp != 1.0:
                        probs = probs ** (1.0 / current_temp)
                    probs = probs / probs.sum()
                    
                    selected_move = np.random.choice(moves, p=probs)
                else:
                    selected_move = best_move
            else:
                # Greedy selection
                selected_move = best_move
            
            if selected_move is None:
                # No move found - shouldn't happen
                winner = None
                break
            
            # Make the move
            board.make_move(selected_move)
            move_count += 1
        
        # Assign values based on game outcome
        values = []
        for i, _ in enumerate(positions):
            # Determine value from perspective of player to move at position i
            position_player = Color.WHITE if i % 2 == 0 else Color.BLACK
            
            if winner is None:
                value = 0.0  # Draw
            elif winner == position_player:
                value = 1.0  # Win
            else:
                value = -1.0  # Loss
            
            values.append(value)
        
        return GameResult(
            positions=positions,
            move_probabilities=move_probs,
            values=values,
            game_length=move_count,
            winner=winner
        )
    
    def _parse_move_string(self, move_str: str) -> Optional[Move]:
        """Parse move string back to Move object (simplified)."""
        # This is a simplified implementation
        # A full version would properly parse the move format
        try:
            # Extract basic move information
            parts = move_str.strip().split()
            if len(parts) >= 4:
                from_sq = int(parts[0]) if parts[0].isdigit() else 0
                to_sq = int(parts[1]) if parts[1].isdigit() else 0
                piece = int(parts[2]) if parts[2].isdigit() else 1
                
                return Move(from_square=from_sq, to_square=to_sq, piece=piece)
        except:
            pass
        return None


class TrainingOrchestrator:
    """
    Main training orchestrator for the chess neural network.
    
    Manages the complete training loop including self-play, data collection,
    and neural network training.
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        
        # Create directories
        Path(config.model_dir).mkdir(exist_ok=True)
        Path(config.data_dir).mkdir(exist_ok=True)
        Path(config.log_dir).mkdir(exist_ok=True)
        
        # Initialize components
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Neural network
        self.network_config = NetworkConfig()
        self.model = AlphaZeroNetwork(self.network_config).to(self.device)
        self.neural_net = NeuralNetworkEvaluator(device=str(self.device))
        self.neural_net.model = self.model
        
        # Chess components
        self.move_generator = MoveGenerator()
        
        # MCTS configuration for self-play
        self.mcts_config = MCTSConfig()
        self.mcts_config.max_simulations = config.mcts_simulations
        self.mcts_config.add_dirichlet_noise = True
        
        # Self-play engine
        self.self_play_engine = SelfPlayEngine(
            self.neural_net, self.move_generator, self.mcts_config
        )
        
        # Training data
        self.training_dataset = TrainingDataset([], [], [])
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config.lr_decay_steps,
            gamma=config.lr_decay_rate
        )
        
        # Training state
        self.iteration = 0
        self.total_games = 0
        self.training_history = []
    
    def train(self):
        """Run the complete training loop."""
        print("Starting AlphaZero-style training...")
        print(f"Configuration: {asdict(self.config)}")
        
        for iteration in range(self.config.num_iterations):
            self.iteration = iteration
            print(f"\n=== Training Iteration {iteration + 1}/{self.config.num_iterations} ===")
            
            # Self-play phase
            print("Generating self-play games...")
            self._generate_self_play_games()
            
            # Training phase
            print("Training neural network...")
            training_loss = self._train_network()
            
            # Logging
            self._log_iteration(training_loss)
            
            # Checkpointing
            if (iteration + 1) % self.config.save_frequency == 0:
                self._save_checkpoint()
            
            # Evaluation
            if (iteration + 1) % self.config.eval_frequency == 0:
                self._evaluate_model()
        
        print("Training completed!")
        self._save_checkpoint(final=True)
    
    def _generate_self_play_games(self):
        """Generate self-play games for training data."""
        games_played = 0
        
        for game_idx in range(self.config.games_per_iteration):
            print(f"  Playing game {game_idx + 1}/{self.config.games_per_iteration}...", 
                  end=' ', flush=True)
            
            try:
                # Play a self-play game
                game_result = self.self_play_engine.play_game(
                    temperature_moves=self.config.temperature_moves,
                    temperature=self.config.temperature
                )
                
                # Add to training data
                self.training_dataset.add_game_data(game_result)
                
                print(f"({game_result.game_length} moves, winner: {game_result.winner})")
                
                games_played += 1
                self.total_games += 1
                
            except Exception as e:
                print(f"Error in game {game_idx}: {e}")
                continue
        
        # Trim dataset to maximum size
        self.training_dataset.trim_to_size(self.config.max_training_samples)
        
        print(f"Generated {games_played} games. Total training samples: {len(self.training_dataset)}")
    
    def _train_network(self) -> float:
        """Train the neural network on collected data."""
        if len(self.training_dataset) < self.config.batch_size:
            print("Not enough training data yet.")
            return 0.0
        
        # Create data loader
        dataloader = DataLoader(
            self.training_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True
        )
        
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(self.config.training_epochs):
            epoch_loss = 0.0
            batch_count = 0
            
            for batch_idx, (positions, policies, values) in enumerate(dataloader):
                positions = positions.to(self.device)
                policies = policies.to(self.device)
                values = values.to(self.device)
                
                # Forward pass
                policy_pred, value_pred = self.model(positions)
                
                # Compute losses
                policy_loss = F.cross_entropy(policy_pred, policies.argmax(dim=1))
                value_loss = F.mse_loss(value_pred.squeeze(), values)
                
                # Combined loss
                loss = policy_loss + value_loss
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
            
            total_loss += epoch_loss / max(batch_count, 1)
            num_batches += 1
            
            if epoch % 2 == 0:
                print(f"    Epoch {epoch + 1}/{self.config.training_epochs}, "
                      f"Loss: {epoch_loss / max(batch_count, 1):.4f}")
        
        # Update learning rate
        self.lr_scheduler.step()
        
        return total_loss / max(num_batches, 1)
    
    def _log_iteration(self, training_loss: float):
        """Log training progress."""
        log_entry = {
            'iteration': self.iteration + 1,
            'total_games': self.total_games,
            'training_samples': len(self.training_dataset),
            'training_loss': training_loss,
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'timestamp': time.time()
        }
        
        self.training_history.append(log_entry)
        
        # Save training log
        log_file = Path(self.config.log_dir) / "training_log.json"
        with open(log_file, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        print(f"Training loss: {training_loss:.4f}, "
              f"LR: {log_entry['learning_rate']:.6f}, "
              f"Samples: {log_entry['training_samples']}")
    
    def _save_checkpoint(self, final: bool = False):
        """Save model checkpoint."""
        checkpoint_name = f"model_iteration_{self.iteration + 1}.pt"
        if final:
            checkpoint_name = "final_model.pt"
        
        checkpoint_path = Path(self.config.model_dir) / checkpoint_name
        
        torch.save({
            'iteration': self.iteration + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.state_dict(),
            'config': asdict(self.config),
            'network_config': asdict(self.network_config),
            'total_games': self.total_games,
            'training_history': self.training_history
        }, checkpoint_path)
        
        print(f"Saved checkpoint: {checkpoint_path}")
        
        # Also save as latest model
        latest_path = Path(self.config.model_dir) / "latest_model.pt"
        torch.save({
            'iteration': self.iteration + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.state_dict(),
            'config': asdict(self.config),
            'network_config': asdict(self.network_config),
            'total_games': self.total_games,
            'training_history': self.training_history
        }, latest_path)
    
    def _evaluate_model(self):
        """Evaluate the current model."""
        print("Evaluating model...")
        
        # Quick evaluation on starting position
        board = ChessBoard()
        position_value = self.neural_net.evaluate_position(board)
        move_probs = self.neural_net.get_move_probabilities(board)
        
        print(f"  Starting position value: {position_value:.3f}")
        print(f"  Number of move probabilities: {len(move_probs)}")
        
        # Could add more sophisticated evaluation here
        # such as playing against a baseline engine
    
    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """Load a training checkpoint."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            self.iteration = checkpoint.get('iteration', 0) - 1
            self.total_games = checkpoint.get('total_games', 0)
            self.training_history = checkpoint.get('training_history', [])
            
            print(f"Loaded checkpoint from iteration {self.iteration + 1}")
            return True
            
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            return False


def start_training(config_file: Optional[str] = None):
    """
    Start the training process.
    
    Args:
        config_file: Optional path to configuration file
    """
    # Load configuration
    if config_file and Path(config_file).exists():
        with open(config_file, 'r') as f:
            config_dict = json.load(f)
        config = TrainingConfig(**config_dict)
    else:
        config = TrainingConfig()
    
    # Create training orchestrator
    trainer = TrainingOrchestrator(config)
    
    # Check for existing checkpoint
    latest_checkpoint = Path(config.model_dir) / "latest_model.pt"
    if latest_checkpoint.exists():
        print("Found existing checkpoint. Loading...")
        trainer.load_checkpoint(str(latest_checkpoint))
    
    # Start training
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        trainer._save_checkpoint()
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise


# Example usage and testing
if __name__ == "__main__":
    print("Testing self-play training infrastructure...")
    
    # Create a minimal configuration for testing
    test_config = TrainingConfig(
        num_iterations=2,
        games_per_iteration=2,
        training_epochs=2,
        mcts_simulations=50,  # Reduced for testing
        batch_size=8
    )
    
    print("Test configuration:")
    for key, value in asdict(test_config).items():
        print(f"  {key}: {value}")
    
    # Test with minimal training
    try:
        trainer = TrainingOrchestrator(test_config)
        print("\nTesting self-play game generation...")
        
        # Generate one test game
        game_result = trainer.self_play_engine.play_game(
            temperature_moves=5, temperature=1.0
        )
        
        print(f"Test game completed:")
        print(f"  Positions: {len(game_result.positions)}")
        print(f"  Game length: {game_result.game_length}")
        print(f"  Winner: {game_result.winner}")
        
        print("Self-play training infrastructure test completed!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
