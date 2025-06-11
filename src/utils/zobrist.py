"""
Zobrist Hashing for Chess
========================

Implementation of Zobrist hashing for fast position identification
in transposition tables and repetition detection.
"""

import random
from typing import List, Dict
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from core.board import Piece, Color


class ZobristHasher:
    """Zobrist hash calculator for chess positions."""
    
    def __init__(self, seed: int = 12345):
        """Initialize Zobrist hash tables with random numbers."""
        random.seed(seed)  # For reproducible hashes
        
        # Hash values for each piece on each square
        self.piece_square_hashes: List[List[List[int]]] = []
        
        # Initialize piece-square hash table [piece][color][square]
        for piece in range(7):  # 0=empty, 1-6=pieces
            color_hashes = []
            for color in range(2):  # 0=white, 1=black
                square_hashes = []
                for square in range(64):
                    square_hashes.append(random.getrandbits(64))
                color_hashes.append(square_hashes)
            self.piece_square_hashes.append(color_hashes)
        
        # Castling rights hashes [white_kingside, white_queenside, black_kingside, black_queenside]
        self.castling_hashes = [random.getrandbits(64) for _ in range(4)]
        
        # En passant file hashes (files a-h)
        self.en_passant_hashes = [random.getrandbits(64) for _ in range(8)]
        
        # Side to move hash (black to move)
        self.side_to_move_hash = random.getrandbits(64)
    
    def hash_position(self, board) -> int:
        """Calculate Zobrist hash for the current position."""
        hash_value = 0
        
        # Hash pieces on squares
        for square in range(64):
            piece, color = board.get_piece_at(square)
            if piece != Piece.EMPTY:
                hash_value ^= self.piece_square_hashes[piece][color][square]
        
        # Hash castling rights
        if board.castling_rights.white_kingside:
            hash_value ^= self.castling_hashes[0]
        if board.castling_rights.white_queenside:
            hash_value ^= self.castling_hashes[1]
        if board.castling_rights.black_kingside:
            hash_value ^= self.castling_hashes[2]
        if board.castling_rights.black_queenside:
            hash_value ^= self.castling_hashes[3]
        
        # Hash en passant target
        if board.en_passant_target is not None:
            file = board.en_passant_target % 8
            hash_value ^= self.en_passant_hashes[file]
        
        # Hash side to move
        if board.turn == Color.BLACK:
            hash_value ^= self.side_to_move_hash
        
        return hash_value
    
    def update_hash_move(self, current_hash: int, move, board_before, board_after) -> int:
        """
        Incrementally update hash after a move.
        This is much faster than recalculating the entire hash.
        """
        hash_value = current_hash
        
        # Remove piece from source square
        piece, color = board_before.get_piece_at(move.from_square)
        hash_value ^= self.piece_square_hashes[piece][color][move.from_square]
        
        # Add piece to destination square
        final_piece = move.promotion_piece if move.promotion_piece != Piece.EMPTY else piece
        hash_value ^= self.piece_square_hashes[final_piece][color][move.to_square]
        
        # Handle capture
        if move.captured_piece != Piece.EMPTY:
            captured_square = move.to_square
            if move.is_en_passant:
                # En passant capture is on a different square
                captured_square = move.to_square + (8 if color == Color.WHITE else -8)
            
            opponent_color = 1 - color
            hash_value ^= self.piece_square_hashes[move.captured_piece][opponent_color][captured_square]
        
        # Handle castling
        if move.is_castling:
            if move.to_square == 62:  # White kingside
                # Move rook from h1 to f1
                hash_value ^= self.piece_square_hashes[Piece.ROOK][Color.WHITE][63]
                hash_value ^= self.piece_square_hashes[Piece.ROOK][Color.WHITE][61]
            elif move.to_square == 58:  # White queenside
                # Move rook from a1 to d1
                hash_value ^= self.piece_square_hashes[Piece.ROOK][Color.WHITE][56]
                hash_value ^= self.piece_square_hashes[Piece.ROOK][Color.WHITE][59]
            elif move.to_square == 6:  # Black kingside
                # Move rook from h8 to f8
                hash_value ^= self.piece_square_hashes[Piece.ROOK][Color.BLACK][7]
                hash_value ^= self.piece_square_hashes[Piece.ROOK][Color.BLACK][5]
            elif move.to_square == 2:  # Black queenside
                # Move rook from a8 to d8
                hash_value ^= self.piece_square_hashes[Piece.ROOK][Color.BLACK][0]
                hash_value ^= self.piece_square_hashes[Piece.ROOK][Color.BLACK][3]
        
        # Update castling rights
        rights_before = board_before.castling_rights
        rights_after = board_after.castling_rights
        
        if rights_before.white_kingside != rights_after.white_kingside:
            hash_value ^= self.castling_hashes[0]
        if rights_before.white_queenside != rights_after.white_queenside:
            hash_value ^= self.castling_hashes[1]
        if rights_before.black_kingside != rights_after.black_kingside:
            hash_value ^= self.castling_hashes[2]
        if rights_before.black_queenside != rights_after.black_queenside:
            hash_value ^= self.castling_hashes[3]
        
        # Update en passant
        if board_before.en_passant_target is not None:
            file = board_before.en_passant_target % 8
            hash_value ^= self.en_passant_hashes[file]
        
        if board_after.en_passant_target is not None:
            file = board_after.en_passant_target % 8
            hash_value ^= self.en_passant_hashes[file]
        
        # Toggle side to move
        hash_value ^= self.side_to_move_hash
        
        return hash_value


# Global instance for use throughout the engine
zobrist_hasher = ZobristHasher()
