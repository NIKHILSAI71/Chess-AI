"""
Opening Book Implementation
==========================

Support for opening book formats including Polyglot and custom formats.
Provides opening move selection with variety and learning capabilities.
"""

import os
import struct
import random
from typing import Optional, List, Dict, Tuple
from pathlib import Path
from dataclasses import dataclass
from abc import ABC, abstractmethod

from core.board import ChessBoard, Move
from utils.zobrist import ZobristHasher


@dataclass
class BookEntry:
    """Opening book entry containing move and metadata."""
    move: Move
    weight: int = 1
    learn: int = 0
    win_count: int = 0
    loss_count: int = 0
    draw_count: int = 0
    
    @property
    def total_games(self) -> int:
        """Total number of games for this move."""
        return self.win_count + self.loss_count + self.draw_count
    
    @property
    def score(self) -> float:
        """Score based on game results."""
        if self.total_games == 0:
            return 0.5
        return (self.win_count + 0.5 * self.draw_count) / self.total_games


class OpeningBook(ABC):
    """Abstract base class for opening books."""
    
    @abstractmethod
    def probe(self, board: ChessBoard) -> Optional[List[BookEntry]]:
        """Probe the book for moves in the current position."""
        pass
    
    @abstractmethod
    def get_book_move(self, board: ChessBoard, variety: float = 0.1) -> Optional[Move]:
        """Get a move from the opening book."""
        pass


class PolyglotBook(OpeningBook):
    """
    Polyglot opening book implementation.
    
    Supports the standard Polyglot .bin format used by many chess engines.
    """
    
    def __init__(self, book_path: Optional[str] = None):
        """Initialize Polyglot book."""
        self.book_path = book_path
        self.book_data: Dict[int, List[BookEntry]] = {}
        self.zobrist = ZobristHasher()
        
        if book_path and os.path.exists(book_path):
            self._load_book()
    
    def _load_book(self):
        """Load Polyglot book from file."""
        if not self.book_path or not os.path.exists(self.book_path):
            return
        
        try:
            with open(self.book_path, 'rb') as f:
                while True:
                    data = f.read(16)  # Polyglot entry is 16 bytes
                    if len(data) < 16:
                        break
                    
                    # Unpack Polyglot entry: key (8 bytes), move (2 bytes), weight (2 bytes), learn (4 bytes)
                    key, move_data, weight, learn = struct.unpack('>QHHl', data)
                    
                    # Convert Polyglot move format to our move format
                    move = self._polyglot_move_to_move(move_data)
                    if move:
                        entry = BookEntry(move=move, weight=weight, learn=learn)
                        
                        if key not in self.book_data:
                            self.book_data[key] = []
                        self.book_data[key].append(entry)
            
            print(f"Loaded opening book with {len(self.book_data)} positions")
            
        except Exception as e:
            print(f"Error loading opening book: {e}")
    
    def _polyglot_move_to_move(self, move_data: int) -> Optional[Move]:
        """Convert Polyglot move format to Move object."""
        try:
            # Polyglot move format (16 bits):
            # Bits 0-5: from square (0-63)
            # Bits 6-11: to square (0-63)  
            # Bits 12-14: promotion piece (0=none, 1=knight, 2=bishop, 3=rook, 4=queen)
            
            from_square = move_data & 0x3F
            to_square = (move_data >> 6) & 0x3F
            promotion = (move_data >> 12) & 0x07
            
            # Convert promotion code
            promotion_piece = None
            if promotion == 1:
                promotion_piece = 'n'
            elif promotion == 2:
                promotion_piece = 'b'
            elif promotion == 3:
                promotion_piece = 'r'
            elif promotion == 4:
                promotion_piece = 'q'
            
            return Move(from_square, to_square, promotion_piece)
            
        except Exception:
            return None
    
    def _get_polyglot_key(self, board: ChessBoard) -> int:
        """Get Polyglot key for the current position."""
        # Use our Zobrist hash as approximation
        # In a full implementation, would need exact Polyglot key calculation
        return self.zobrist.hash_position(board)
    
    def probe(self, board: ChessBoard) -> Optional[List[BookEntry]]:
        """Probe the book for moves in the current position."""
        key = self._get_polyglot_key(board)
        return self.book_data.get(key)
    
    def get_book_move(self, board: ChessBoard, variety: float = 0.1) -> Optional[Move]:
        """
        Get a move from the opening book.
        
        Args:
            board: Current board position
            variety: Randomness factor (0.0 = always best, 1.0 = completely random)
        
        Returns:        Move from book or None if position not in book
        """
        entries = self.probe(board)
        if not entries:
            return None
        
        # Filter for legal moves
        from core.moves import MoveGenerator
        move_gen = MoveGenerator()
        legal_moves = set(str(move) for move in move_gen.generate_legal_moves(board))
        valid_entries = [entry for entry in entries if str(entry.move) in legal_moves]
        
        if not valid_entries:
            return None
        
        if variety <= 0.0:
            # Always return the move with highest weight
            return max(valid_entries, key=lambda e: e.weight).move
        
        if variety >= 1.0:
            # Completely random selection
            return random.choice(valid_entries).move
        
        # Weighted random selection based on variety
        total_weight = sum(entry.weight for entry in valid_entries)
        if total_weight == 0:
            return random.choice(valid_entries).move
        
        # Apply variety factor to weights
        adjusted_weights = []
        for entry in valid_entries:
            base_prob = entry.weight / total_weight
            # Mix with uniform distribution based on variety
            adjusted_prob = (1 - variety) * base_prob + variety * (1 / len(valid_entries))
            adjusted_weights.append(adjusted_prob)
        
        # Select based on adjusted probabilities
        rand_val = random.random()
        cumulative = 0.0
        
        for i, prob in enumerate(adjusted_weights):
            cumulative += prob
            if rand_val <= cumulative:
                return valid_entries[i].move
        
        # Fallback
        return valid_entries[-1].move
    
    def update_result(self, board: ChessBoard, move: Move, result: str):
        """
        Update book statistics based on game result.
        
        Args:
            board: Position where move was played
            move: Move that was played
            result: Game result ('1-0', '0-1', '1/2-1/2')
        """
        key = self._get_polyglot_key(board)
        entries = self.book_data.get(key)
        
        if not entries:
            return
        
        # Find the matching entry
        for entry in entries:
            if entry.move == move:
                if result == '1-0':
                    if board.current_player.name.lower() == 'white':
                        entry.win_count += 1
                    else:
                        entry.loss_count += 1
                elif result == '0-1':
                    if board.current_player.name.lower() == 'black':
                        entry.win_count += 1
                    else:
                        entry.loss_count += 1
                elif result == '1/2-1/2':
                    entry.draw_count += 1
                break


class CustomOpeningBook(OpeningBook):
    """
    Custom opening book with learning capabilities.
    
    Stores positions and moves in a more flexible format than Polyglot.
    """
    
    def __init__(self, book_path: Optional[str] = None):
        """Initialize custom opening book."""
        self.book_path = book_path
        self.positions: Dict[str, List[BookEntry]] = {}
        self.move_history: List[Tuple[str, Move, str]] = []  # (position_hash, move, result)
        
        if book_path and os.path.exists(book_path):
            self._load_book()
    
    def _load_book(self):
        """Load custom book from JSON format."""
        try:
            import json
            with open(self.book_path, 'r') as f:
                data = json.load(f)
                
            for pos_hash, entries_data in data.get('positions', {}).items():
                entries = []
                for entry_data in entries_data:
                    move = Move(
                        entry_data['from_square'],
                        entry_data['to_square'],
                        entry_data.get('promotion')
                    )
                    entry = BookEntry(
                        move=move,
                        weight=entry_data.get('weight', 1),
                        win_count=entry_data.get('win_count', 0),
                        loss_count=entry_data.get('loss_count', 0),
                        draw_count=entry_data.get('draw_count', 0)
                    )
                    entries.append(entry)
                self.positions[pos_hash] = entries
                
            print(f"Loaded custom opening book with {len(self.positions)} positions")
            
        except Exception as e:
            print(f"Error loading custom opening book: {e}")
    
    def save_book(self):
        """Save custom book to file."""
        if not self.book_path:
            return
        
        try:
            import json
            data = {
                'positions': {},
                'metadata': {
                    'format_version': '1.0',
                    'created_by': 'Chess-AI Engine'
                }
            }
            
            for pos_hash, entries in self.positions.items():
                entries_data = []
                for entry in entries:
                    entry_data = {
                        'from_square': entry.move.from_square,
                        'to_square': entry.move.to_square,
                        'weight': entry.weight,
                        'win_count': entry.win_count,
                        'loss_count': entry.loss_count,
                        'draw_count': entry.draw_count
                    }
                    if entry.move.promotion:
                        entry_data['promotion'] = entry.move.promotion
                    entries_data.append(entry_data)
                data['positions'][pos_hash] = entries_data
            
            with open(self.book_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            print(f"Error saving custom opening book: {e}")
    
    def probe(self, board: ChessBoard) -> Optional[List[BookEntry]]:
        """Probe the book for moves in the current position."""
        pos_hash = self._get_position_hash(board)
        return self.positions.get(pos_hash)
    
    def get_book_move(self, board: ChessBoard, variety: float = 0.1) -> Optional[Move]:
        """Get a move from the opening book."""
        entries = self.probe(board)
        if not entries:
            return None
          # Filter for legal moves
        from core.moves import MoveGenerator
        move_gen = MoveGenerator()
        legal_moves = set(str(move) for move in move_gen.generate_legal_moves(board))
        valid_entries = [entry for entry in entries if str(entry.move) in legal_moves]
        
        if not valid_entries:
            return None
        
        # Use score-based selection
        if variety <= 0.0:
            return max(valid_entries, key=lambda e: e.score).move
        
        # Weighted selection based on scores
        scores = [entry.score for entry in valid_entries]
        if all(score == 0.5 for score in scores):
            # All moves have equal score, use weights
            weights = [entry.weight for entry in valid_entries]
            return random.choices(valid_entries, weights=weights)[0].move
        
        # Apply variety to score-based selection
        adjusted_scores = []
        for score in scores:
            # Mix score with uniform distribution
            adjusted_score = (1 - variety) * score + variety * 0.5
            adjusted_scores.append(adjusted_score)
        
        return random.choices(valid_entries, weights=adjusted_scores)[0].move
    
    def add_position(self, board: ChessBoard, move: Move, weight: int = 1):
        """Add a position and move to the book."""
        pos_hash = self._get_position_hash(board)
        
        if pos_hash not in self.positions:
            self.positions[pos_hash] = []
        
        # Check if move already exists
        for entry in self.positions[pos_hash]:
            if entry.move == move:
                entry.weight += weight
                return
        
        # Add new entry
        entry = BookEntry(move=move, weight=weight)
        self.positions[pos_hash].append(entry)
    
    def _get_position_hash(self, board: ChessBoard) -> str:
        """Get a hash string for the position."""
        # Use FEN without move counters for position identity
        fen_parts = board.to_fen().split()
        position_fen = ' '.join(fen_parts[:4])  # Board, side, castling, en passant
        return position_fen


def create_opening_book(book_path: Optional[str] = None, book_type: str = 'polyglot') -> OpeningBook:
    """
    Create an opening book instance.
    
    Args:
        book_path: Path to book file
        book_type: Type of book ('polyglot' or 'custom')
    
    Returns:
        OpeningBook instance
    """
    if book_type.lower() == 'polyglot':
        return PolyglotBook(book_path)
    elif book_type.lower() == 'custom':
        return CustomOpeningBook(book_path)
    else:
        raise ValueError(f"Unknown book type: {book_type}")
