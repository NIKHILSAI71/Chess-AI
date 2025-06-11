#!/usr/bin/env python3
"""
Advanced Training Data Management System
=======================================

Implements sophisticated data acquisition, curation, and augmentation pipelines
for training world-class chess neural networks.
"""

import torch
import torch.utils.data as data
import numpy as np
import pandas as pd
import chess
import chess.pgn
import json
import os
import random
from typing import List, Tuple, Dict, Optional, Iterator
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import pickle
import logging
from pathlib import Path

from ..core.board import ChessBoard, Move, Piece, Color
from ..neural.network import BoardEncoder, MoveEncoder
from ..utils.fen import parse_fen

logger = logging.getLogger(__name__)

@dataclass
class TrainingExample:
    """A single training example with board state and target values."""
    board_state: np.ndarray  # Encoded board position
    move_probabilities: np.ndarray  # Target move probability distribution
    value: float  # Target position value (-1 to 1)
    game_outcome: int  # Final game result: 1=white wins, 0=draw, -1=black wins
    fen: str  # Original FEN string for debugging
    move_san: str  # Move in SAN notation
    depth_from_end: int  # How many moves until game end

@dataclass
class DatasetConfig:
    """Configuration for training data generation."""
    # Data sources
    pgn_files: List[str] = None
    position_files: List[str] = None
    tactical_puzzle_files: List[str] = None
    
    # Filtering criteria
    min_elo: int = 2000  # Minimum player ELO for human games
    max_moves: int = 200  # Maximum game length
    min_moves: int = 10   # Minimum game length
    
    # Data augmentation
    use_symmetries: bool = True  # Board rotations/reflections
    noise_factor: float = 0.1    # Random noise for exploration
    
    # Self-play integration
    self_play_ratio: float = 0.3  # Ratio of self-play vs. human games
    
    # Processing
    batch_size: int = 256
    num_workers: int = 8
    cache_size: int = 100000  # Number of positions to cache in memory

class ChessDatasetManager:
    """Advanced chess dataset management with multiple data sources."""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.board_encoder = BoardEncoder()
        self.move_encoder = MoveEncoder()
        self.examples_cache = []
        self.cache_size = config.cache_size
        
    def load_pgn_games(self, pgn_file: str) -> Iterator[TrainingExample]:
        """Load and process games from PGN files."""
        logger.info(f"Loading PGN file: {pgn_file}")
        
        with open(pgn_file, 'r', encoding='utf-8', errors='ignore') as f:
            game_count = 0
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                
                try:
                    # Filter by ELO if available
                    white_elo = game.headers.get('WhiteElo', '0')
                    black_elo = game.headers.get('BlackElo', '0')
                    
                    if white_elo.isdigit() and black_elo.isdigit():
                        if int(white_elo) < self.config.min_elo or int(black_elo) < self.config.min_elo:
                            continue
                    
                    # Extract training examples from game
                    examples = self._extract_examples_from_game(game)
                    for example in examples:
                        yield example
                        
                    game_count += 1
                    if game_count % 100 == 0:
                        logger.info(f"Processed {game_count} games from {pgn_file}")
                        
                except Exception as e:
                    logger.warning(f"Error processing game: {e}")
                    continue
    
    def _extract_examples_from_game(self, game) -> List[TrainingExample]:
        """Extract training examples from a single game."""
        examples = []
        board = chess.Board()
        moves = list(game.mainline_moves())
        
        # Determine game outcome
        result = game.headers.get('Result', '*')
        if result == '1-0':
            game_outcome = 1  # White wins
        elif result == '0-1':
            game_outcome = -1  # Black wins
        else:
            game_outcome = 0  # Draw
        
        # Filter by game length
        if len(moves) < self.config.min_moves or len(moves) > self.config.max_moves:
            return examples
        
        # Process each position in the game
        for move_idx, move in enumerate(moves):
            try:
                # Convert position to our format
                chess_board = self._chess_board_to_our_board(board)
                
                # Encode board state
                board_state = self.board_encoder.encode_board(chess_board)
                
                # Generate move probability distribution
                # In supervised learning, we set the played move to high probability
                move_probs = np.zeros(4096)  # Assuming 4096 possible moves
                move_idx_encoded = self.move_encoder.encode_move(move, board)
                if move_idx_encoded is not None:
                    move_probs[move_idx_encoded] = 1.0
                
                # Calculate position value based on game outcome and position
                depth_from_end = len(moves) - move_idx
                position_value = self._calculate_position_value(
                    game_outcome, depth_from_end, board.turn
                )
                
                example = TrainingExample(
                    board_state=board_state,
                    move_probabilities=move_probs,
                    value=position_value,
                    game_outcome=game_outcome,
                    fen=board.fen(),
                    move_san=board.san(move),
                    depth_from_end=depth_from_end
                )
                
                examples.append(example)
                
                # Apply data augmentation if enabled
                if self.config.use_symmetries:
                    augmented = self._augment_example(example)
                    examples.extend(augmented)
                
                # Make the move
                board.push(move)
                
            except Exception as e:
                logger.warning(f"Error processing move {move_idx}: {e}")
                continue
        
        return examples
    
    def _chess_board_to_our_board(self, chess_board: chess.Board) -> ChessBoard:
        """Convert python-chess board to our ChessBoard format."""
        # This is a simplified conversion - you may need to enhance this
        board = ChessBoard.from_fen(chess_board.fen())
        return board
    
    def _calculate_position_value(self, game_outcome: int, depth_from_end: int, 
                                  turn: bool) -> float:
        """Calculate position value based on game outcome and position."""
        # Base value from game outcome
        base_value = float(game_outcome)
        
        # Adjust for player turn (white=True, black=False)
        if not turn:  # Black to move
            base_value = -base_value
        
        # Apply temporal discount - positions closer to end are more certain
        discount = 1.0 - (depth_from_end / 200.0)  # Gradual discount
        discount = max(0.1, min(1.0, discount))
        
        return base_value * discount
    
    def _augment_example(self, example: TrainingExample) -> List[TrainingExample]:
        """Apply data augmentation techniques."""
        augmented = []
        
        # Board symmetries (horizontal flip)
        if self.config.use_symmetries and random.random() < 0.5:
            # Implement board flipping logic
            # This requires careful handling of piece positions and move encoding
            pass
        
        # Add noise for exploration (during self-play)
        if hasattr(self.config, 'add_noise') and self.config.add_noise:
            noisy_probs = example.move_probabilities.copy()
            noise = np.random.dirichlet([0.3] * len(noisy_probs)) * self.config.noise_factor
            noisy_probs = noisy_probs * (1 - self.config.noise_factor) + noise
            
            augmented_example = TrainingExample(
                board_state=example.board_state,
                move_probabilities=noisy_probs,
                value=example.value,
                game_outcome=example.game_outcome,
                fen=example.fen,
                move_san=example.move_san,
                depth_from_end=example.depth_from_end
            )
            augmented.append(augmented_example)
        
        return augmented
    
    def load_tactical_puzzles(self, puzzle_file: str) -> Iterator[TrainingExample]:
        """Load tactical puzzles for specialized training."""
        logger.info(f"Loading tactical puzzles from: {puzzle_file}")
        
        # Implement puzzle loading - format depends on your puzzle dataset
        # Common formats: CSV with FEN, solution moves, and difficulty ratings
        pass
    
    def save_processed_data(self, examples: List[TrainingExample], output_file: str):
        """Save processed training examples to disk."""
        logger.info(f"Saving {len(examples)} examples to {output_file}")
        
        with open(output_file, 'wb') as f:
            pickle.dump(examples, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    def load_processed_data(self, input_file: str) -> List[TrainingExample]:
        """Load pre-processed training examples."""
        logger.info(f"Loading examples from {input_file}")
        
        with open(input_file, 'rb') as f:
            examples = pickle.load(f)
        
        logger.info(f"Loaded {len(examples)} examples")
        return examples

class ChessDataset(data.Dataset):
    """PyTorch dataset for chess training data."""
    
    def __init__(self, examples: List[TrainingExample], transform=None):
        self.examples = examples
        self.transform = transform
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Convert to tensors
        board_state = torch.FloatTensor(example.board_state)
        move_probs = torch.FloatTensor(example.move_probabilities)
        value = torch.FloatTensor([example.value])
        
        if self.transform:
            board_state = self.transform(board_state)
        
        return {
            'board_state': board_state,
            'move_probabilities': move_probs,
            'value': value,
            'game_outcome': example.game_outcome
        }

def create_data_loaders(config: DatasetConfig, train_split: float = 0.8) -> Tuple[data.DataLoader, data.DataLoader]:
    """Create training and validation data loaders."""
    manager = ChessDatasetManager(config)
    
    # Load all training examples
    all_examples = []
    
    # Load from PGN files
    if config.pgn_files:
        for pgn_file in config.pgn_files:
            if os.path.exists(pgn_file):
                examples = list(manager.load_pgn_games(pgn_file))
                all_examples.extend(examples)
    
    # Load tactical puzzles
    if config.tactical_puzzle_files:
        for puzzle_file in config.tactical_puzzle_files:
            if os.path.exists(puzzle_file):
                examples = list(manager.load_tactical_puzzles(puzzle_file))
                all_examples.extend(examples)
    
    # Shuffle and split
    random.shuffle(all_examples)
    split_idx = int(len(all_examples) * train_split)
    
    train_examples = all_examples[:split_idx]
    val_examples = all_examples[split_idx:]
    
    # Create datasets
    train_dataset = ChessDataset(train_examples)
    val_dataset = ChessDataset(val_examples)
    
    # Create data loaders
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    logger.info(f"Created data loaders: {len(train_examples)} train, {len(val_examples)} val")
    
    return train_loader, val_loader

if __name__ == "__main__":
    # Example usage
    config = DatasetConfig(
        pgn_files=['trainning-data/Chess-game-data.csv'],  # Adjust path as needed
        min_elo=2200,
        batch_size=128,
        use_symmetries=True
    )
    
    train_loader, val_loader = create_data_loaders(config)
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
