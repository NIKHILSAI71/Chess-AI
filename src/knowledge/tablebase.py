"""
Endgame Tablebase Implementation
===============================

Support for Syzygy and other endgame tablebase formats.
Provides perfect endgame play and accurate evaluation.
"""

import os
import sys
from typing import Optional, Dict, List, Tuple
from enum import Enum
from pathlib import Path
from abc import ABC, abstractmethod

from core.board import ChessBoard, Move, Color


class TablebaseResult(Enum):
    """Tablebase lookup result."""
    WIN = "win"
    LOSS = "loss" 
    DRAW = "draw"
    CURSED_WIN = "cursed_win"  # Win but 50-move rule
    BLESSED_LOSS = "blessed_loss"  # Loss but 50-move rule
    UNKNOWN = "unknown"


class TablebaseProber(ABC):
    """Abstract base class for tablebase probers."""
    
    @abstractmethod
    def probe_wdl(self, board: ChessBoard) -> Tuple[TablebaseResult, int]:
        """
        Probe for Win/Draw/Loss result.
        
        Returns:
            Tuple of (result, distance_to_mate_or_zero)
        """
        pass
    
    @abstractmethod
    def probe_dtm(self, board: ChessBoard) -> Optional[int]:
        """
        Probe for Distance to Mate.
        
        Returns:
            Distance to mate in plies, or None if not available
        """
        pass
    
    @abstractmethod
    def probe_dtz(self, board: ChessBoard) -> Optional[int]:
        """
        Probe for Distance to Zero (fifty-move rule reset).
        
        Returns:
            Distance to fifty-move rule reset, or None if not available
        """
        pass
    
    @abstractmethod
    def get_best_move(self, board: ChessBoard) -> Optional[Move]:
        """Get the best move according to tablebase."""
        pass


class SyzygyTablebase(TablebaseProber):
    """
    Syzygy tablebase implementation.
    
    Supports up to 7-piece Syzygy tablebases with WDL and DTZ information.
    """
    
    def __init__(self, tablebase_path: Optional[str] = None):
        """Initialize Syzygy tablebase."""
        self.tablebase_path = tablebase_path
        self.available_pieces = set()
        self.syzygy_available = False
        
        # Try to import python-chess syzygy module
        try:
            import chess
            import chess.syzygy
            self.chess = chess
            self.syzygy = chess.syzygy
            self.syzygy_available = True
            
            if tablebase_path and os.path.exists(tablebase_path):
                self._scan_available_tablebases()
                
        except ImportError:
            print("python-chess with Syzygy support not available")
            print("Install with: pip install python-chess[engine]")
    
    def _scan_available_tablebases(self):
        """Scan for available tablebase files."""
        if not self.tablebase_path:
            return
        
        try:
            for file in os.listdir(self.tablebase_path):
                if file.endswith('.rtbw') or file.endswith('.rtbz'):
                    # Extract piece count from filename
                    pieces = len(file.split('.')[0])
                    self.available_pieces.add(pieces)
            
            if self.available_pieces:
                max_pieces = max(self.available_pieces)
                print(f"Syzygy tablebase available up to {max_pieces} pieces")
                
        except Exception as e:
            print(f"Error scanning tablebase directory: {e}")
    
    def _board_to_chess_board(self, board: ChessBoard):
        """Convert our board to python-chess board."""
        if not self.syzygy_available:
            return None
        
        try:
            fen = board.to_fen()
            return self.chess.Board(fen)
        except Exception:
            return None
    
    def is_available(self, board: ChessBoard) -> bool:
        """Check if tablebase is available for this position."""
        if not self.syzygy_available or not self.tablebase_path:
            return False
        
        piece_count = bin(board.all_pieces).count('1')
        return piece_count in self.available_pieces and piece_count <= 7
    
    def probe_wdl(self, board: ChessBoard) -> Tuple[TablebaseResult, int]:
        """Probe for Win/Draw/Loss result."""
        if not self.is_available(board):
            return TablebaseResult.UNKNOWN, 0
        
        try:
            chess_board = self._board_to_chess_board(board)
            if not chess_board:
                return TablebaseResult.UNKNOWN, 0
            
            with self.syzygy.open_tablebase(self.tablebase_path) as tablebase:
                wdl = tablebase.probe_wdl(chess_board)
                
                if wdl is None:
                    return TablebaseResult.UNKNOWN, 0
                elif wdl > 0:
                    return TablebaseResult.WIN, abs(wdl)
                elif wdl < 0:
                    return TablebaseResult.LOSS, abs(wdl)
                else:
                    return TablebaseResult.DRAW, 0
                    
        except Exception as e:
            print(f"Tablebase probe error: {e}")
            return TablebaseResult.UNKNOWN, 0
    
    def probe_dtm(self, board: ChessBoard) -> Optional[int]:
        """Probe for Distance to Mate."""
        # DTM not available in Syzygy format
        return None
    
    def probe_dtz(self, board: ChessBoard) -> Optional[int]:
        """Probe for Distance to Zero."""
        if not self.is_available(board):
            return None
        
        try:
            chess_board = self._board_to_chess_board(board)
            if not chess_board:
                return None
            
            with self.syzygy.open_tablebase(self.tablebase_path) as tablebase:
                dtz = tablebase.probe_dtz(chess_board)
                return dtz
                
        except Exception as e:
            print(f"Tablebase DTZ probe error: {e}")
            return None
    
    def get_best_move(self, board: ChessBoard) -> Optional[Move]:
        """Get the best move according to tablebase."""
        if not self.is_available(board):
            return None
        
        try:
            chess_board = self._board_to_chess_board(board)
            if not chess_board:
                return None
            
            with self.syzygy.open_tablebase(self.tablebase_path) as tablebase:
                # Get current position result
                current_wdl = tablebase.probe_wdl(chess_board)
                if current_wdl is None:
                    return None
                
                best_move = None
                best_wdl = None
                
                # Try all legal moves
                for chess_move in chess_board.legal_moves:
                    chess_board.push(chess_move)
                    
                    # Get result after move (negated for opponent)
                    move_wdl = tablebase.probe_wdl(chess_board)
                    if move_wdl is not None:
                        move_wdl = -move_wdl
                        
                        if best_wdl is None or move_wdl > best_wdl:
                            best_wdl = move_wdl
                            best_move = chess_move
                    
                    chess_board.pop()
                
                if best_move:
                    # Convert back to our move format
                    return Move(
                        best_move.from_square,
                        best_move.to_square,
                        best_move.promotion.symbol().lower() if best_move.promotion else None
                    )
                    
        except Exception as e:
            print(f"Tablebase best move error: {e}")
        
        return None


class MockTablebase(TablebaseProber):
    """
    Mock tablebase for testing and fallback.
    
    Provides basic endgame knowledge without external dependencies.
    """
    
    def __init__(self):
        """Initialize mock tablebase."""
        self.endgame_knowledge = {
            # King vs King
            ('K', 'k'): TablebaseResult.DRAW,
            # King and Queen vs King
            ('KQ', 'k'): TablebaseResult.WIN,
            ('K', 'kq'): TablebaseResult.LOSS,
            # King and Rook vs King
            ('KR', 'k'): TablebaseResult.WIN,
            ('K', 'kr'): TablebaseResult.LOSS,
            # King and Pawn vs King (simplified)
            ('KP', 'k'): TablebaseResult.WIN,  # Usually winning
            ('K', 'kp'): TablebaseResult.LOSS,
        }
    
    def _get_material_signature(self, board: ChessBoard) -> Tuple[str, str]:
        """Get material signature for both sides."""
        white_pieces = []
        black_pieces = []
          # Count pieces (simplified - would need actual piece counting)
        piece_count = bin(board.occupied.value).count('1')
        
        # This is a simplified version - real implementation would
        # extract actual piece types from the board
        if piece_count <= 4:
            return ('K', 'k')  # Simplified for demo
        else:
            return ('', '')  # Unknown endgame
    
    def probe_wdl(self, board: ChessBoard) -> Tuple[TablebaseResult, int]:
        """Probe for Win/Draw/Loss result."""
        white_sig, black_sig = self._get_material_signature(board)
        
        if board.to_move == Color.WHITE:
            key = (white_sig, black_sig)
        else:
            key = (black_sig, white_sig)
            
        result = self.endgame_knowledge.get(key, TablebaseResult.UNKNOWN)
        
        # Estimate distance (simplified)
        if result in [TablebaseResult.WIN, TablebaseResult.LOSS]:
            return result, 50  # Estimate 50 moves to mate
        else:
            return result, 0
    
    def probe_dtm(self, board: ChessBoard) -> Optional[int]:
        """Probe for Distance to Mate."""
        result, distance = self.probe_wdl(board)
        if result in [TablebaseResult.WIN, TablebaseResult.LOSS]:
            return distance
        return None
    
    def probe_dtz(self, board: ChessBoard) -> Optional[int]:
        """Probe for Distance to Zero."""
        # Simplified implementation
        return self.probe_dtm(board)
    
    def get_best_move(self, board: ChessBoard) -> Optional[Move]:
        """Get the best move according to tablebase."""
        # Simplified - just return first legal move in winning positions
        result, _ = self.probe_wdl(board)
        if result == TablebaseResult.WIN:
            legal_moves = board.get_legal_moves()
            return legal_moves[0] if legal_moves else None
        return None


def create_tablebase(tablebase_path: Optional[str] = None, 
                    tablebase_type: str = 'syzygy') -> TablebaseProber:
    """
    Create a tablebase prober instance.
    
    Args:
        tablebase_path: Path to tablebase files
        tablebase_type: Type of tablebase ('syzygy' or 'mock')
    
    Returns:
        TablebaseProber instance
    """
    if tablebase_type.lower() == 'syzygy':
        return SyzygyTablebase(tablebase_path)
    elif tablebase_type.lower() == 'mock':
        return MockTablebase()
    else:
        raise ValueError(f"Unknown tablebase type: {tablebase_type}")


class TablebaseManager:
    """
    Manager for multiple tablebase sources.
    
    Coordinates between different tablebase formats and provides
    unified access to endgame knowledge.
    """
    
    def __init__(self):
        """Initialize tablebase manager."""
        self.probers: List[TablebaseProber] = []
        self.cache: Dict[str, Tuple[TablebaseResult, int]] = {}
        self.max_cache_size = 10000
    
    def add_prober(self, prober: TablebaseProber):
        """Add a tablebase prober."""
        self.probers.append(prober)
    
    def probe_position(self, board: ChessBoard) -> Tuple[TablebaseResult, int]:
        """
        Probe position using all available tablebases.
        
        Returns the first successful result found.
        """
        # Check cache first
        fen = board.to_fen()
        if fen in self.cache:
            return self.cache[fen]
        
        # Try each prober
        for prober in self.probers:
            result, distance = prober.probe_wdl(board)
            if result != TablebaseResult.UNKNOWN:
                # Cache result
                if len(self.cache) < self.max_cache_size:
                    self.cache[fen] = (result, distance)
                return (result, distance)
        
        return (TablebaseResult.UNKNOWN, 0)
    
    def get_best_move(self, board: ChessBoard) -> Optional[Move]:
        """Get best move from any available tablebase."""
        for prober in self.probers:
            move = prober.get_best_move(board)
            if move:
                return move
        return None
    
    def is_tablebase_position(self, board: ChessBoard) -> bool:
        """Check if position can be found in any tablebase."""
        result, _ = self.probe_position(board)
        return result != TablebaseResult.UNKNOWN
