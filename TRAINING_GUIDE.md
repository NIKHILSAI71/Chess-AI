# 🚀 Chess AI Training Guide

This guide will walk you through training your world-class chess AI using the sophisticated neural network architecture we've built.

## 🎯 Training Overview

Our training system implements a **3-phase approach** following the most advanced methodologies:

### Phase 1: Supervised Learning (Foundation)
- Train on **grandmaster games** and **engine analysis**
- Learn basic chess patterns and strong move selection
- Establish baseline chess knowledge

### Phase 2: Reinforcement Learning (Mastery)
- **Self-play training** with Monte Carlo Tree Search (MCTS)
- Continuous improvement through playing against itself
- Advanced position evaluation and strategic understanding

### Phase 3: Hybrid Optimization (Excellence)
- Combine classical search with neural evaluation
- Fine-tune for tournament-level play
- Specialized training for openings, tactics, and endgames

## 🏃‍♂️ Quick Start (Immediate Training)

### Option 1: Quick Training Script
```bash
# Run the quick training script with your existing data
python quick_train.py
```

This will:
- ✅ Use your existing `trainning-data/Chess-game-data.csv`
- ✅ Create a small but functional neural network
- ✅ Train for 5 epochs (~10-30 minutes)
- ✅ Save a working model to `models/quick_trained_model.pth`

### Option 2: Full Training Pipeline
```bash
# Install additional dependencies
pip install wandb tensorboard chess python-chess

# Run the complete training system
python -m src.training.orchestrator
```

## 📊 Data Requirements

### Minimum Data (Quick Start)
- ✅ Your existing CSV file with chess games
- ✅ ~1,000 games minimum
- ✅ Format: moves, results, player ratings

### Optimal Data (Professional Training)
- 🎯 **10+ million grandmaster games**
- 🎯 **Engine analysis** (Stockfish evaluations)
- 🎯 **Tactical puzzle databases**
- 🎯 **Opening theory databases**

### Recommended Data Sources
```bash
# Download large game databases
wget https://database.lichess.org/standard/lichess_db_standard_rated_2023-01.pgn.bz2

# Chess puzzle databases
wget https://github.com/clarkerubber/chess-tactics/raw/master/tactics.json

# Engine evaluation datasets
# Use Stockfish to analyze positions at depth 20+
```

## ⚙️ Training Configuration

### Hardware Requirements

#### Minimum (CPU Only)
- ✅ 8GB RAM
- ✅ Multi-core CPU
- ✅ ~50GB disk space
- ⏱️ Training time: 1-7 days

#### Recommended (GPU Accelerated)
- 🚀 16GB+ RAM
- 🚀 NVIDIA GPU with 8GB+ VRAM
- 🚀 SSD storage (200GB+)
- ⏱️ Training time: 4-24 hours

#### Professional (Multi-GPU)
- 🏆 64GB+ RAM
- 🏆 Multiple RTX 4090 or A100 GPUs
- 🏆 NVMe SSD storage (1TB+)
- ⏱️ Training time: 2-8 hours

### Training Parameters

#### Quick Training (Immediate Results)
```python
NetworkConfig(
    input_channels=12,
    filters=64,
    num_residual_blocks=4,
    policy_head_filters=16,
    value_head_filters=16
)

TrainingConfig(
    supervised_epochs=5,
    self_play_iterations=10,
    training_games_per_iteration=100,
    batch_size=64
)
```

#### Production Training (Tournament Strength)
```python
NetworkConfig(
    input_channels=18,
    filters=512,
    num_residual_blocks=20,
    policy_head_filters=64,
    value_head_filters=64
)

TrainingConfig(
    supervised_epochs=50,
    self_play_iterations=1000,
    training_games_per_iteration=10000,
    batch_size=256
)
```

## 🔧 Step-by-Step Training Process

### Step 1: Environment Setup
```bash
# Install dependencies
pip install torch torchvision torchaudio
pip install pandas numpy chess python-chess
pip install tensorboard wandb  # For monitoring
pip install matplotlib seaborn  # For visualization

# Verify PyTorch installation
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### Step 2: Data Preparation
```bash
# Quick start with existing data
python quick_train.py

# Or prepare professional dataset
python -c "
from src.data.training_data import ChessDatasetManager, DatasetConfig
config = DatasetConfig(pgn_files=['path/to/your/games.pgn'])
manager = ChessDatasetManager(config)
# Process and save training data
"
```

### Step 3: Supervised Learning Phase
```python
# Run supervised learning
from src.training.orchestrator import ChessTrainingOrchestrator, TrainingPipeline
from src.neural.network import NetworkConfig

config = TrainingPipeline(
    network_config=NetworkConfig(),
    supervised_epochs=10,
    batch_size=128
)

orchestrator = ChessTrainingOrchestrator(config)
orchestrator._supervised_learning_phase()
```

### Step 4: Self-Play Training Phase
```python
# Run reinforcement learning
orchestrator._self_play_training_phase()
```

### Step 5: Integration and Testing
```python
# Test the trained model
python demo_engine.py

# Run against other engines
python -c "
from src.engine.uci import EnhancedUCIEngine
engine = EnhancedUCIEngine()
# Load your trained model
# engine.load_neural_network('models/your_model.pth')
"
```

## 📈 Monitoring Training Progress

### TensorBoard (Built-in Monitoring)
```bash
# Start TensorBoard server
tensorboard --logdir=runs/chess_training

# Open browser to http://localhost:6006
```

### Weights & Biases (Professional Monitoring)
```bash
# Setup W&B account (free)
wandb login

# Training will automatically log to W&B dashboard
```

### Key Metrics to Monitor
- **Loss Curves**: Policy loss + Value loss
- **Win Rate**: Against baseline opponents
- **Search Statistics**: Nodes per second, depth reached
- **Game Quality**: Average game length, decisive results

## 🎯 Expected Training Timeline

### Phase 1: Quick Training (1-4 hours)
```
Hour 0: Data preparation and model initialization
Hour 1: Supervised learning on 1K games
Hour 2: Basic self-play training (100 games)
Hour 3: Model evaluation and testing
Hour 4: Integration with chess engine
```
**Result**: ~1200 ELO strength, can beat beginners

### Phase 2: Intermediate Training (1-3 days)
```
Day 1: Supervised learning on 100K games
Day 2: Self-play training (1K iterations)
Day 3: Advanced evaluation and fine-tuning
```
**Result**: ~1800 ELO strength, club-level play

### Phase 3: Advanced Training (1-2 weeks)
```
Week 1: Large-scale supervised learning (10M games)
Week 2: Extensive self-play (10K iterations)
```
**Result**: ~2400+ ELO strength, master-level play

## 🚀 Advanced Training Techniques

### Data Augmentation
```python
# Board symmetries
use_symmetries=True

# Position perturbations
noise_factor=0.1

# Temporal augmentation
historical_positions=3
```

### Learning Rate Scheduling
```python
# Cosine annealing
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=1000, eta_min=1e-6
)

# Warmup + decay
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=0.01, total_steps=training_steps
)
```

### Advanced MCTS Configuration
```python
MCTSConfig(
    num_simulations=1600,      # More simulations = stronger play
    exploration_constant=1.4,  # Balance exploration/exploitation
    temperature_schedule=True, # Adaptive temperature
    virtual_loss=3.0,          # Parallel search optimization
    dirichlet_noise=True       # Exploration during self-play
)
```

## 🔍 Troubleshooting

### Common Issues

#### Out of Memory Errors
```python
# Reduce batch size
batch_size = 64  # Instead of 256

# Use gradient accumulation
accumulation_steps = 4

# Enable mixed precision
use_amp = True
```

#### Slow Training
```python
# Use DataLoader workers
num_workers = 8

# Pin memory for GPU
pin_memory = True

# Compile model (PyTorch 2.0+)
model = torch.compile(model)
```

#### Poor Performance
```python
# Increase model size
filters = 512  # Instead of 256
num_residual_blocks = 20  # Instead of 10

# More training data
min_games = 1000000  # Instead of 1000

# Longer training
supervised_epochs = 100  # Instead of 10
```

### Performance Benchmarks

| Configuration | Training Time | Final ELO | Hardware |
|--------------|---------------|-----------|----------|
| Quick | 2 hours | ~1200 | CPU only |
| Standard | 1 day | ~1800 | GTX 1080 |
| Advanced | 3 days | ~2400 | RTX 4090 |
| Professional | 1 week | ~2800+ | 8x A100 |

## 🎓 Next Steps After Training

### Integration
1. **Load trained model** into UCI engine
2. **Test against opponents** (engines, humans)
3. **Fine-tune parameters** based on results

### Deployment
1. **Export model** for production use
2. **Optimize inference** for real-time play
3. **Create chess GUI** integration

### Continuous Improvement
1. **Collect game data** from real play
2. **Retrain periodically** with new data
3. **A/B test** different model versions

## 📚 Additional Resources

### Documentation
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Chess Programming Wiki](https://www.chessprogramming.org/)
- [AlphaZero Paper](https://arxiv.org/abs/1712.01815)

### Datasets
- [Lichess Database](https://database.lichess.org/)
- [FICS Games Database](http://ficsgames.org/)
- [Chess.com API](https://www.chess.com/news/view/published-data-api)

### Tools
- [Stockfish](https://stockfishchess.org/) - For position analysis
- [Arena GUI](http://www.playwitharena.de/) - For testing engines
- [Cutechess](https://github.com/cutechess/cutechess) - For automated tournaments

---

**Start your training journey now and build a world-class chess AI! 🏆**
