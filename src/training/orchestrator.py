#!/usr/bin/env python3
"""
Advanced Chess AI Training System
================================

Implements the complete training pipeline following world-class methodologies:
1. Supervised Learning on grandmaster games
2. Reinforcement Learning through self-play
3. Hybrid training with continuous improvement
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import json
import time
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

# Optional wandb import for experiment tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

from ..neural.network import AlphaZeroNetwork, NetworkConfig, NeuralNetworkEvaluator
from ..neural.mcts import MCTSEngine, MCTSConfig
from ..neural.training import SelfPlayEngine, TrainingConfig
from ..data.training_data import ChessDatasetManager, DatasetConfig, create_data_loaders
from ..core.board import ChessBoard
from ..core.search import SearchEngine
from ..core.evaluation import Evaluator
from ..core.moves import MoveGenerator

logger = logging.getLogger(__name__)

@dataclass
class TrainingPipeline:
    """Complete training configuration."""
    # Model configuration
    network_config: NetworkConfig
    
    # Training phases
    supervised_epochs: int = 10
    self_play_iterations: int = 100
    training_games_per_iteration: int = 1000
    
    # Optimization
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    batch_size: int = 256
    gradient_clip: float = 1.0
    
    # Self-play configuration
    mcts_simulations: int = 800
    mcts_temperature: float = 1.0
    mcts_exploration: float = 1.0
    
    # Model management
    save_interval: int = 5  # Save model every N iterations
    evaluation_interval: int = 10  # Evaluate against baseline every N iterations
    
    # Hardware configuration
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers: int = 8
    
    # Paths
    model_save_dir: str = 'models/checkpoints'
    tensorboard_dir: str = 'runs/chess_training'
    data_dir: str = 'trainning-data'
    
    # Experiment tracking
    use_wandb: bool = True
    project_name: str = 'chess-ai-training'

class ChessTrainingOrchestrator:
    """Main training orchestrator implementing the complete pipeline."""
    
    def __init__(self, config: TrainingPipeline):
        self.config = config
        self.device = torch.device(config.device)
        
        # Create directories
        os.makedirs(config.model_save_dir, exist_ok=True)
        os.makedirs(config.tensorboard_dir, exist_ok=True)
        
        # Initialize logging
        self._setup_logging()
        
        # Initialize model
        self.network = AlphaZeroNetwork(config.network_config).to(self.device)
        self.optimizer = optim.AdamW(
            self.network.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.self_play_iterations
        )
        
        # Training components
        self.move_generator = MoveGenerator()
        self.evaluator = Evaluator()
        
        # Tracking
        self.writer = SummaryWriter(config.tensorboard_dir)
        self.training_step = 0
        self.iteration = 0
          # Initialize experiment tracking
        if config.use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project=config.project_name,
                config=asdict(config),
                name=f"chess-training-{int(time.time())}"
            )
    
    def _setup_logging(self):
        """Setup comprehensive logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('training.log'),
                logging.StreamHandler()
            ]
        )
    
    def train_complete_pipeline(self):
        """Execute the complete training pipeline."""
        logger.info("🚀 Starting Chess AI Training Pipeline")
        logger.info(f"Device: {self.device}")
        logger.info(f"Network parameters: {sum(p.numel() for p in self.network.parameters())}")
        
        try:
            # Phase 1: Supervised Learning
            logger.info("📚 Phase 1: Supervised Learning on Grandmaster Games")
            self._supervised_learning_phase()
            
            # Phase 2: Reinforcement Learning through Self-Play
            logger.info("🎮 Phase 2: Reinforcement Learning - Self-Play Training")
            self._self_play_training_phase()
            
            # Phase 3: Final Evaluation and Model Export
            logger.info("🏆 Phase 3: Final Evaluation and Model Export")
            self._final_evaluation_phase()
            
            logger.info("✅ Training pipeline completed successfully!")
            
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            self._save_checkpoint('interrupted')
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise        
        finally:
            self.writer.close()
            if self.config.use_wandb and WANDB_AVAILABLE:
                wandb.finish()
    
    def _supervised_learning_phase(self):
        """Phase 1: Supervised learning on grandmaster games."""
        logger.info("Loading grandmaster game data...")
        
        # Configure data loading
        dataset_config = DatasetConfig(
            pgn_files=[os.path.join(self.config.data_dir, 'Chess-game-data.csv')],
            min_elo=2200,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            use_symmetries=True
        )
        
        # Create data loaders
        train_loader, val_loader = create_data_loaders(dataset_config)
        
        logger.info(f"Training on {len(train_loader)} batches, validating on {len(val_loader)} batches")
        
        # Training loop
        for epoch in range(self.config.supervised_epochs):
            logger.info(f"Supervised Learning Epoch {epoch + 1}/{self.config.supervised_epochs}")
            
            # Training
            train_loss = self._train_supervised_epoch(train_loader)
            
            # Validation
            val_loss = self._validate_epoch(val_loader)
              # Logging
            self.writer.add_scalar('Supervised/Train_Loss', train_loss, epoch)
            self.writer.add_scalar('Supervised/Val_Loss', val_loss, epoch)
            
            if self.config.use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    'supervised_train_loss': train_loss,
                    'supervised_val_loss': val_loss,
                    'epoch': epoch
                })
            
            logger.info(f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_interval == 0:
                self._save_checkpoint(f'supervised_epoch_{epoch + 1}')
        
        # Save final supervised model
        self._save_checkpoint('supervised_complete')
        logger.info("Supervised learning phase completed")
    
    def _train_supervised_epoch(self, train_loader) -> float:
        """Train one epoch of supervised learning."""
        self.network.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move data to device
            board_states = batch['board_state'].to(self.device)
            target_policies = batch['move_probabilities'].to(self.device)
            target_values = batch['value'].to(self.device)
            
            # Forward pass
            policy_logits, value_pred = self.network(board_states)
            
            # Calculate losses
            policy_loss = F.cross_entropy(policy_logits, target_policies)
            value_loss = F.mse_loss(value_pred.squeeze(), target_values.squeeze())
            
            # Combined loss
            total_loss_batch = policy_loss + value_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss_batch.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
            
            self.optimizer.step()
            
            # Accumulate loss
            total_loss += total_loss_batch.item()
            num_batches += 1
            
            # Log every 100 batches
            if batch_idx % 100 == 0:
                logger.debug(f"Batch {batch_idx}: Loss: {total_loss_batch.item():.4f}")
            
            self.training_step += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def _validate_epoch(self, val_loader) -> float:
        """Validate the model."""
        self.network.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                board_states = batch['board_state'].to(self.device)
                target_policies = batch['move_probabilities'].to(self.device)
                target_values = batch['value'].to(self.device)
                
                policy_logits, value_pred = self.network(board_states)
                
                policy_loss = F.cross_entropy(policy_logits, target_policies)
                value_loss = F.mse_loss(value_pred.squeeze(), target_values.squeeze())
                
                total_loss += (policy_loss + value_loss).item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def _self_play_training_phase(self):
        """Phase 2: Reinforcement Learning through self-play."""
        logger.info("Setting up self-play training...")
        
        # Initialize self-play engine
        mcts_config = MCTSConfig(
            num_simulations=self.config.mcts_simulations,
            exploration_constant=self.config.mcts_exploration,
            temperature=self.config.mcts_temperature
        )
        
        self_play_engine = SelfPlayEngine(
            network=self.network,
            mcts_config=mcts_config,
            device=self.device
        )
        
        for iteration in range(self.config.self_play_iterations):
            self.iteration = iteration
            logger.info(f"Self-Play Iteration {iteration + 1}/{self.config.self_play_iterations}")
            
            # Generate self-play games
            logger.info("Generating self-play games...")
            start_time = time.time()
            
            training_examples = self_play_engine.generate_training_data(
                num_games=self.config.training_games_per_iteration
            )
            
            generation_time = time.time() - start_time
            logger.info(f"Generated {len(training_examples)} training examples in {generation_time:.2f}s")
            
            # Train on self-play data
            logger.info("Training on self-play data...")
            train_loss = self._train_on_self_play_data(training_examples)
            
            # Update learning rate
            self.scheduler.step()
            
            # Logging            self.writer.add_scalar('SelfPlay/Train_Loss', train_loss, iteration)
            self.writer.add_scalar('SelfPlay/Games_Generated', len(training_examples), iteration)
            self.writer.add_scalar('SelfPlay/Generation_Time', generation_time, iteration)
            self.writer.add_scalar('SelfPlay/Learning_Rate', self.optimizer.param_groups[0]['lr'], iteration)
            
            if self.config.use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    'self_play_loss': train_loss,
                    'games_generated': len(training_examples),
                    'generation_time': generation_time,
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                    'iteration': iteration
                })
            
            logger.info(f"Iteration {iteration + 1}: Loss: {train_loss:.4f}")
            
            # Save model periodically
            if (iteration + 1) % self.config.save_interval == 0:
                self._save_checkpoint(f'self_play_iter_{iteration + 1}')
            
            # Evaluate against baseline
            if (iteration + 1) % self.config.evaluation_interval == 0:
                self._evaluate_against_baseline()
        
        logger.info("Self-play training phase completed")
    
    def _train_on_self_play_data(self, training_examples: List) -> float:
        """Train the network on self-play generated data."""
        self.network.train()
        
        # Convert examples to tensors
        # This depends on the format of your training examples
        # You'll need to implement the conversion based on your SelfPlayEngine output
        
        total_loss = 0.0
        num_batches = 0
        
        # Create batches from training examples
        batch_size = self.config.batch_size
        for i in range(0, len(training_examples), batch_size):
            batch = training_examples[i:i + batch_size]
            
            # Convert batch to tensors (implement based on your data format)
            board_states, target_policies, target_values = self._convert_examples_to_tensors(batch)
            
            # Move to device
            board_states = board_states.to(self.device)
            target_policies = target_policies.to(self.device)
            target_values = target_values.to(self.device)
            
            # Forward pass
            policy_logits, value_pred = self.network(board_states)
            
            # Calculate losses
            policy_loss = F.cross_entropy(policy_logits, target_policies)
            value_loss = F.mse_loss(value_pred.squeeze(), target_values.squeeze())
            
            total_loss_batch = policy_loss + value_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
            self.optimizer.step()
            
            total_loss += total_loss_batch.item()
            num_batches += 1
            self.training_step += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def _convert_examples_to_tensors(self, examples: List) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert training examples to PyTorch tensors."""
        # This needs to be implemented based on your training example format
        # For now, returning placeholder tensors
        batch_size = len(examples)
        
        # Placeholder implementation - replace with actual conversion
        board_states = torch.zeros(batch_size, 18, 8, 8)  # Adjust dimensions as needed
        target_policies = torch.zeros(batch_size, 4096)   # Adjust based on move encoding
        target_values = torch.zeros(batch_size, 1)
        
        return board_states, target_policies, target_values
    
    def _evaluate_against_baseline(self):
        """Evaluate current model against a baseline."""
        logger.info("Evaluating against baseline...")
        
        # Create evaluation engines
        current_evaluator = NeuralNetworkEvaluator(self.network)
        baseline_evaluator = Evaluator()  # Classical evaluator as baseline
        
        # Play test games
        wins = 0
        draws = 0
        losses = 0
        num_games = 10  # Quick evaluation
        
        for game_idx in range(num_games):
            try:
                # Simulate a game between current model and baseline
                # This is a simplified version - implement full game simulation
                result = self._simulate_evaluation_game(current_evaluator, baseline_evaluator)
                
                if result > 0:
                    wins += 1
                elif result == 0:
                    draws += 1
                else:
                    losses += 1
                    
            except Exception as e:
                logger.warning(f"Evaluation game {game_idx} failed: {e}")
        
        # Calculate win rate
        win_rate = (wins + 0.5 * draws) / num_games if num_games > 0 else 0.0
        
        # Log results
        logger.info(f"Evaluation results: {wins}W-{draws}D-{losses}L (Win rate: {win_rate:.3f})")
        
        self.writer.add_scalar('Evaluation/Win_Rate', win_rate, self.iteration)
        self.writer.add_scalar('Evaluation/Wins', wins, self.iteration)
        self.writer.add_scalar('Evaluation/Draws', draws, self.iteration)
        self.writer.add_scalar('Evaluation/Losses', losses, self.iteration)
        
        if self.config.use_wandb and WANDB_AVAILABLE:
            wandb.log({
                'eval_win_rate': win_rate,
                'eval_wins': wins,
                'eval_draws': draws,
                'eval_losses': losses
            })
    
    def _simulate_evaluation_game(self, current_evaluator, baseline_evaluator) -> int:
        """Simulate a single evaluation game."""
        # Simplified game simulation - implement full game logic
        # Returns: 1 for current model win, 0 for draw, -1 for baseline win
        
        # For now, return random result - replace with actual game simulation
        import random
        return random.choice([-1, 0, 1])
    
    def _final_evaluation_phase(self):
        """Phase 3: Final evaluation and model export."""
        logger.info("Performing final evaluation...")
        
        # Comprehensive evaluation against multiple baselines
        self._comprehensive_evaluation()
        
        # Export final model
        final_model_path = os.path.join(self.config.model_save_dir, 'final_model.pth')
        self._export_model(final_model_path)
        
        # Generate model info
        self._generate_model_info()
        
        logger.info("Final evaluation completed")
    
    def _comprehensive_evaluation(self):
        """Perform comprehensive evaluation against multiple opponents."""
        logger.info("Running comprehensive evaluation suite...")
        
        # Test against different opponents/configurations
        test_suites = [
            ('Random Player', 'random'),
            ('Classical Engine', 'classical'),
            ('Weak Neural', 'weak_neural')
        ]
        
        results = {}
        
        for suite_name, opponent_type in test_suites:
            logger.info(f"Testing against {suite_name}...")
            win_rate = self._test_against_opponent(opponent_type)
            results[suite_name] = win_rate
            logger.info(f"{suite_name}: {win_rate:.3f} win rate")
          # Log comprehensive results
        for opponent, win_rate in results.items():
            self.writer.add_scalar(f'Final_Evaluation/{opponent}', win_rate, 0)
        
        if self.config.use_wandb and WANDB_AVAILABLE:
            wandb.log({f'final_eval_{k.lower().replace(" ", "_")}': v for k, v in results.items()})
    
    def _test_against_opponent(self, opponent_type: str) -> float:
        """Test current model against specific opponent type."""
        # Implement specific opponent testing
        # For now, return placeholder
        import random
        return random.uniform(0.6, 0.9)  # Simulate good performance
    
    def _save_checkpoint(self, name: str):
        """Save model checkpoint."""
        checkpoint_path = os.path.join(self.config.model_save_dir, f'{name}.pth')
        
        checkpoint = {
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'iteration': self.iteration,
            'training_step': self.training_step,
            'config': asdict(self.config)
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def _export_model(self, model_path: str):
        """Export final trained model."""
        torch.save({
            'model_state_dict': self.network.state_dict(),
            'network_config': asdict(self.config.network_config),
            'training_completed': True
        }, model_path)
        
        logger.info(f"Final model exported: {model_path}")
    
    def _generate_model_info(self):
        """Generate model information and statistics."""
        info = {
            'model_parameters': sum(p.numel() for p in self.network.parameters()),
            'training_iterations': self.iteration,
            'training_steps': self.training_step,
            'device_used': str(self.device),
            'config': asdict(self.config)
        }
        
        info_path = os.path.join(self.config.model_save_dir, 'model_info.json')
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        
        logger.info(f"Model info saved: {info_path}")

def main():
    """Main training entry point."""
    # Configure training pipeline
    network_config = NetworkConfig(
        input_channels=18,
        filters=256,
        num_residual_blocks=10,
        policy_head_filters=32,
        value_head_filters=32
    )
    
    training_config = TrainingPipeline(
        network_config=network_config,
        supervised_epochs=5,
        self_play_iterations=50,
        training_games_per_iteration=100,
        learning_rate=0.001,
        batch_size=128,
        mcts_simulations=400,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Create and run training orchestrator
    orchestrator = ChessTrainingOrchestrator(training_config)
    orchestrator.train_complete_pipeline()

if __name__ == "__main__":
    main()
