# Chess AI - Grandmaster Level Engine

An ambitious project to create a chess AI capable of competing with grandmasters, following the architectural principles of AlphaZero and Stockfish.

## Project Structure

```
Chess-AI/
├── src/
│   ├── core/                 # Core chess engine
│   │   ├── board.py          # Board representation and state
│   │   ├── moves.py          # Move generation and validation
│   │   ├── evaluation.py     # Position evaluation functions
│   │   └── search.py         # Search algorithms (Alpha-Beta, MCTS)
│   ├── neural/               # Neural network components
│   │   ├── models.py         # Network architectures
│   │   ├── training.py       # Training loops and optimization
│   │   └── inference.py      # NN inference for evaluation
│   ├── data/                 # Data processing and management
│   │   ├── loader.py         # Game data loading and parsing
│   │   ├── preprocessor.py   # Data cleaning and augmentation
│   │   └── generator.py      # Self-play data generation
│   ├── engine/               # UCI protocol and engine interface
│   │   ├── uci.py           # UCI communication protocol
│   │   └── interface.py     # Engine main interface
│   └── utils/               # Utility functions
│       ├── zobrist.py       # Zobrist hashing for transposition tables
│       ├── pgn_parser.py    # PGN file parsing
│       └── fen.py           # FEN notation handling
├── training-data/           # Training datasets
├── models/                  # Saved neural network models
├── tests/                   # Unit tests and validation
├── benchmarks/              # Performance testing
└── config/                  # Configuration files
```

## Phase 1: Foundational Engine (Current)

### Completed:
- [ ] Project structure setup
- [ ] Board representation (bitboards)
- [ ] Move generation engine
- [ ] Alpha-Beta search with basic optimizations
- [ ] Simple evaluation function
- [ ] UCI protocol implementation
- [ ] Initial data analysis and processing

### Technologies:
- **Core Engine**: Python (rapid development) with critical parts in C++ for performance
- **Neural Networks**: PyTorch
- **Data Processing**: Pandas, NumPy
- **Build System**: Poetry for dependency management
- **Version Control**: Git

## Getting Started

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Analyze training data:
```bash
python src/data/loader.py
```

3. Run the engine:
```bash
python src/engine/uci.py
```

## Data Analysis

Current training data: `Chess-game-data.csv` with ~20,000 chess games
- Opening variations, move sequences, game outcomes
- Rich source for initial pattern recognition and evaluation training

## Development Phases

1. **Phase 1**: Core Engine & Data Processing ⚠️ *Current*
2. **Phase 2**: Neural Network Development & Supervised Learning
3. **Phase 3**: Reinforcement Learning & MCTS Implementation
4. **Phase 4**: Knowledge Integration & Performance Optimization
5. **Phase 5**: Testing, Iteration & Grandmaster Challenge

## Contributing

This is an ambitious long-term project. Each component is designed to be modular and extensible for continuous improvement.
