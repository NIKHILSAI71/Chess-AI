# Consolidated Chess AI - Streamlined Structure

## Overview

Your Chess AI project has been reorganized into a clean, streamlined structure with two main consolidated files that handle all core functionality:

1. **`train_ai.py`** - Unified training script
2. **`play/chess_game_server.py`** - Consolidated game backend server

## Quick Start

### 1. Train Your AI (2 minutes)
```bash
# Quick training with default settings
python train_ai.py

# Or use a configuration file
python train_ai.py --config config/quick_train_config.json

# Full production training
python train_ai.py --full
```

### 2. Play Against Your AI
```bash
# Start the game server
cd play
python chess_game_server.py

# Open your browser to: http://localhost:5000
```

## New Consolidated Structure

### Training System (`train_ai.py`)
- **Unified Interface**: Single script for all training needs
- **Multiple Modes**: Quick, production, and custom training
- **Auto-Configuration**: Sensible defaults with easy customization
- **Progress Tracking**: Built-in monitoring and checkpointing
- **Model Management**: Automatic saving and validation

**Features:**
- Loads data from `models/initial_training_positions.json`
- Creates sample data if none exists
- Supports both quick and production training modes
- Integrates with your existing `src/` neural network components
- Saves trained models to `models/production_chess_model.pth`

### Game Backend (`play/chess_game_server.py`)
- **Single Server**: Replaces all previous backend files
- **Multiple AI Engines**: Supports neural networks, MCTS, search engines
- **REST API**: Clean API for frontend integration
- **Dynamic Engine Switching**: Change AI engines on the fly
- **Difficulty Levels**: 5 configurable difficulty levels
- **Game Management**: Handles multiple concurrent games

**Features:**
- Serves the chess game interface at `http://localhost:5000`
- Loads trained models automatically from `models/` directory
- Provides `/api/` endpoints for all game operations
- Supports multiple AI engines (demo, neural, MCTS, search)
- Real-time position analysis and move generation

## Training Configurations

### Quick Training (`config/quick_train_config.json`)
- **Duration**: ~5-10 minutes
- **Purpose**: Fast results for testing
- **Model**: Smaller network (128 filters, 6 residual blocks)
- **Data**: Uses sample data or existing positions

### Production Training (`config/production_train_config.json`)
- **Duration**: Several hours
- **Purpose**: Tournament-strength AI
- **Model**: Large network (256 filters, 12 residual blocks)
- **Data**: Uses full dataset from `trainning-data/Chess-game-data.csv`

### Custom Training
Create your own JSON configuration file with custom parameters.

## API Endpoints

The consolidated server provides these main endpoints:

### Game Management
- `GET /` - Serve chess game interface
- `POST /api/new_game` - Create new game
- `POST /api/make_move` - Process moves and get AI responses

### AI Control
- `GET /api/engines` - List available AI engines
- `POST /api/set_engine` - Switch AI engine
- `POST /api/set_difficulty` - Change difficulty level

### Analysis
- `POST /api/evaluate` - Analyze chess positions
- `GET /api/game_info` - Get server and AI information
- `GET /api/health` - Health check

## File Changes Summary

### New Files Created:
- `train_ai.py` - Consolidated training script
- `play/chess_game_server.py` - Unified backend server
- `config/quick_train_config.json` - Quick training config
- `config/production_train_config.json` - Production training config

### Files That Can Be Removed:
Since their functionality is now consolidated, you can optionally remove:
- `chess_backend.py` → Consolidated into `chess_game_server.py`
- `chess_backend_fixed.py` → Consolidated into `chess_game_server.py`
- `advanced_chess_backend.py` → Consolidated into `chess_game_server.py`
- `simple_backend.py` → Consolidated into `chess_game_server.py`
- Any training demos/scripts → Consolidated into `train_ai.py`

### Files Kept:
- All `src/` directory components (neural networks, core logic)
- `play/index.html` and frontend files (chess-ui.js, chess-logic.js, etc.)
- Your existing models and data
- Documentation files

## Usage Examples

### Training Examples
```bash
# Quick training (5 minutes)
python train_ai.py --quick

# Full production training
python train_ai.py --full

# Custom training with specific data
python train_ai.py --data "path/to/data.json" --output "models/my_model.pth"

# Training with configuration file
python train_ai.py --config config/my_custom_config.json
```

### Playing Examples
```bash
# Start server
cd play
python chess_game_server.py

# The server will automatically:
# - Load any trained models from models/
# - Serve the game at http://localhost:5000
# - Provide multiple AI engines
# - Support difficulty adjustment
```

### AI Engine Switching
The frontend can now switch between different AI engines:
- **Demo**: Random but reasonable moves
- **Neural**: Uses your trained neural networks
- **MCTS**: Monte Carlo Tree Search with neural guidance
- **Search**: Classical alpha-beta search
- **Hybrid**: Combined neural + search approaches

## Benefits of Consolidation

1. **Simplified Structure**: Two main files instead of many scattered scripts
2. **Easy Training**: Single command to train from scratch to production
3. **Unified Backend**: One server handles all game operations
4. **Better Organization**: Clear separation between training and playing
5. **Easier Maintenance**: Fewer files to track and update
6. **Enhanced Features**: More AI engines and configuration options
7. **Better Documentation**: Clear usage patterns and examples

## Next Steps

1. **Test Training**: Run `python train_ai.py` to create your first model
2. **Test Playing**: Start the server and play against your AI
3. **Experiment**: Try different engines and difficulty levels
4. **Customize**: Modify configuration files for your specific needs
5. **Iterate**: Use the training script to continuously improve your AI

Your Chess AI project is now more organized, easier to use, and ready for both quick experimentation and serious development!
