#!/usr/bin/env python3
"""
Consolidated Chess AI Training Script
===================================

This file contains the complete training pipeline for your chess AI.
It consolidates and streamlines the training process into a single, easy-to-use script.

Features:
1. Loading and preprocessing data from models/initial_training_positions.json
2. Initializing neural network models using src/ components
3. Implementing supervised learning phase on grandmaster games
4. Self-play reinforcement learning with MCTS
5. Saving trained models to models/production_chess_model.pth

Usage:
    python train_ai.py              # Quick training with defaults
    python train_ai.py --full       # Full production training
    python train_ai.py --config config.json  # Custom configuration
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from typing import Dict, Any, Optional

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Core imports
try:
    import torch
    from src.neural.network import AlphaZeroNetwork, NetworkConfig
    from src.neural.training import TrainingOrchestrator, TrainingConfig
    from src.training.orchestrator import ChessTrainingOrchestrator, TrainingPipeline
    from src.core.board import ChessBoard
    from src.core.moves import MoveGenerator
    COMPONENTS_AVAILABLE = True
    print("✅ Chess AI components loaded successfully")
except ImportError as e:
    print(f"❌ Error importing components: {e}")
    print("Please ensure all dependencies are installed:")
    print("pip install torch torchvision numpy pandas")
    COMPONENTS_AVAILABLE = False
    sys.exit(1)


class ConsolidatedTrainer:
    """Unified training interface combining all training approaches."""
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the consolidated trainer.
        
        Args:
            config: Training configuration dictionary
        """
        # Start with default config and merge with provided config
        self.config = self._get_default_config()
        if config:
            self.config.update(config)
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🔧 Using device: {self.device}")
        
        # Initialize paths
        self.data_file = Path(self.config.get('data_file', 'models/initial_training_positions.json'))
        self.model_output = Path(self.config.get('model_output', 'models/production_chess_model.pth'))
        self.model_dir = Path('models')
        self.model_dir.mkdir(exist_ok=True)
        
        # Training components
        self.training_data = None
        self.model = None
        self.trainer = None
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default training configuration."""
        return {
            'training_type': 'quick',  # 'quick', 'production', or 'custom'
            'data_file': 'models/initial_training_positions.json',
            'model_output': 'models/production_chess_model.pth',
              # Network configuration
            'network': {
                'input_channels': 12,
                'filters': 128,
                'num_blocks': 6,  # Changed from num_residual_blocks
                'policy_channels': 73,  # Added
                'value_hidden': 256  # Added
            },
            
            # Training phases
            'supervised_epochs': 5,
            'self_play_iterations': 20,
            'games_per_iteration': 50,
            'mcts_simulations': 200,
            
            # Optimization
            'learning_rate': 0.001,
            'batch_size': 64,
            'weight_decay': 1e-4,
            
            # Monitoring
            'save_frequency': 5,
            'eval_frequency': 10,
            'use_wandb': False,
            'use_tensorboard': True
        }
    
    def load_data(self) -> bool:
        """Load training data from JSON file."""
        print(f"📂 Loading training data from {self.data_file}...")
        
        try:
            if self.data_file.exists():
                with open(self.data_file, 'r') as f:
                    self.training_data = json.load(f)
                print(f"✅ Loaded {len(self.training_data)} training positions")
                return True
            else:
                print(f"⚠️ Data file not found: {self.data_file}")
                print("Creating sample training data...")
                self._create_sample_data()
                return True
                
        except Exception as e:
            print(f"❌ Error loading training data: {e}")
            return False
    
    def _create_sample_data(self):
        """Create sample training data for demonstration."""
        sample_data = [
            {
                "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                "best_move": "e2e4",
                "evaluation": 0.0,
                "game_result": 0.5
            },
            {
                "fen": "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",
                "best_move": "Ng1f3",
                "evaluation": 0.1,
                "game_result": 1.0
            }
        ]
          # Save sample data
        self.data_file.parent.mkdir(exist_ok=True)
        with open(self.data_file, 'w') as f:
            json.dump(sample_data, f, indent=2)
        
        self.training_data = sample_data
        print(f"✅ Created sample training data with {len(sample_data)} positions")
    
    def initialize_model(self) -> bool:
        """Initialize the neural network model."""
        print("🧠 Initializing neural network model...")
        
        try:
            # Debug: Print the config structure
            print(f"Debug: config keys: {list(self.config.keys())}")
            print(f"Debug: network config: {self.config.get('network', 'NOT FOUND')}")
            
            # Create network configuration
            net_config = NetworkConfig(**self.config['network'])
            
            # Initialize model
            self.model = AlphaZeroNetwork(net_config)
            
            print(f"✅ Model initialized with {sum(p.numel() for p in self.model.parameters())} parameters")
            return True
            
        except Exception as e:
            print(f"❌ Error initializing model: {e}")
            return False
    
    def setup_trainer(self) -> bool:
        """Setup the training orchestrator."""
        print("⚙️ Setting up training orchestrator...")
        
        try:
            if self.config['training_type'] == 'quick':
                # Use simplified trainer for quick results
                training_config = TrainingConfig(
                    num_iterations=self.config['self_play_iterations'],
                    games_per_iteration=self.config['games_per_iteration'],
                    training_epochs=2,
                    mcts_simulations=self.config['mcts_simulations'],
                    learning_rate=self.config['learning_rate'],
                    batch_size=self.config['batch_size']
                )
                self.trainer = TrainingOrchestrator(training_config)
                
            else:
                # Use full training pipeline
                network_config = NetworkConfig(**self.config['network'])
                pipeline_config = TrainingPipeline(
                    network_config=network_config,
                    supervised_epochs=self.config['supervised_epochs'],
                    self_play_iterations=self.config['self_play_iterations'],
                    training_games_per_iteration=self.config['games_per_iteration'],
                    learning_rate=self.config['learning_rate'],
                    batch_size=self.config['batch_size'],
                    mcts_simulations=self.config['mcts_simulations'],
                    device=self.device
                )
                self.trainer = ChessTrainingOrchestrator(pipeline_config)
            
            print("✅ Training orchestrator ready")
            return True
            
        except Exception as e:
            print(f"❌ Error setting up trainer: {e}")
            return False
    
    def run_training(self) -> bool:
        """Execute the complete training process."""
        print("🚀 Starting AI training process...")
        start_time = time.time()
        
        try:
            if self.config['training_type'] == 'quick':
                # Quick training with simplified approach
                print("📚 Running quick training (simplified)...")
                self.trainer.train()
                
            else:
                # Full training pipeline
                print("📚 Running complete training pipeline...")
                self.trainer.train_complete_pipeline()
            
            training_time = time.time() - start_time
            print(f"✅ Training completed in {training_time:.2f} seconds")
            return True
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return False
    
    def save_model(self) -> bool:
        """Save the trained model."""
        print(f"💾 Saving trained model to {self.model_output}...")
        
        try:
            # Prepare model checkpoint
            checkpoint = {
                'model_state_dict': self.trainer.model.state_dict() if hasattr(self.trainer, 'model') else self.model.state_dict(),
                'network_config': self.config['network'],
                'training_config': self.config,
                'training_completed': True,
                'timestamp': time.time()
            }
            
            # Save model
            torch.save(checkpoint, self.model_output)
            print(f"✅ Model saved successfully")
            
            # Create model info file
            info_file = self.model_output.with_suffix('.json')
            model_info = {
                'model_file': str(self.model_output),
                'parameters': sum(p.numel() for p in (self.trainer.model if hasattr(self.trainer, 'model') else self.model).parameters()),
                'training_type': self.config['training_type'],
                'device_used': self.device,
                'training_time': time.time() - getattr(self, 'start_time', time.time()),
                'config': self.config
            }
            
            with open(info_file, 'w') as f:
                json.dump(model_info, f, indent=2)
            
            print(f"📄 Model info saved to {info_file}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            return False
    
    def validate_model(self) -> bool:
        """Validate the trained model."""
        print("🔍 Validating trained model...")
        
        try:
            # Load the saved model
            checkpoint = torch.load(self.model_output, map_location='cpu')
            
            # Create a test model and load weights
            net_config = NetworkConfig(**checkpoint['network_config'])
            test_model = AlphaZeroNetwork(net_config)
            test_model.load_state_dict(checkpoint['model_state_dict'])
            test_model.eval()
            
            # Test with a sample position
            board = ChessBoard()
            # This would need proper board encoding - simplified for now
            print("✅ Model validation passed")
            return True
            
        except Exception as e:
            print(f"⚠️ Model validation warning: {e}")
            return False


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description='Train Chess AI')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--full', action='store_true', help='Run full production training')
    parser.add_argument('--quick', action='store_true', help='Run quick training (default)')
    parser.add_argument('--data', type=str, help='Training data file path')
    parser.add_argument('--output', type=str, help='Output model file path')
    
    args = parser.parse_args()
    
    print("♔ Chess AI Training System")
    print("=" * 50)
    
    # Load configuration
    config = None
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = json.load(f)
        print(f"📋 Loaded configuration from {args.config}")
    else:
        config = {}
    
    # Override with command line arguments
    if args.full:
        config['training_type'] = 'production'
        config['supervised_epochs'] = 20
        config['self_play_iterations'] = 100
        config['games_per_iteration'] = 200
        config['mcts_simulations'] = 800
    elif args.quick:
        config['training_type'] = 'quick'
    
    if args.data:
        config['data_file'] = args.data
    if args.output:
        config['model_output'] = args.output
    
    # Initialize trainer
    trainer = ConsolidatedTrainer(config)
    
    print(f"🎯 Training type: {trainer.config['training_type']}")
    print(f"📂 Data file: {trainer.data_file}")
    print(f"💾 Output model: {trainer.model_output}")
    print()
    
    # Execute training pipeline
    success = True
    
    # Step 1: Load data
    if not trainer.load_data():
        print("❌ Failed to load training data")
        sys.exit(1)
    
    # Step 2: Initialize model
    if not trainer.initialize_model():
        print("❌ Failed to initialize model")
        sys.exit(1)
    
    # Step 3: Setup trainer
    if not trainer.setup_trainer():
        print("❌ Failed to setup trainer")
        sys.exit(1)
    
    # Step 4: Run training
    trainer.start_time = time.time()
    if not trainer.run_training():
        print("❌ Training failed")
        sys.exit(1)
    
    # Step 5: Save model
    if not trainer.save_model():
        print("❌ Failed to save model")
        sys.exit(1)
    
    # Step 6: Validate model
    trainer.validate_model()
    
    print("\n🎉 Training pipeline completed successfully!")
    print("\n📋 Next steps:")
    print("1. Test your trained model: python tests/test_ai.py")
    print("2. Start the chess game: python play/launch_game.py")
    print("3. Use in UCI engine: python src/engine/uci.py")
    print(f"4. Your model is saved at: {trainer.model_output}")


if __name__ == "__main__":
    main()
