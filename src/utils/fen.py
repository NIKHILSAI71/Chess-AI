"""
FEN (Forsyth-Edwards Notation) Parser
====================================

Utilities for parsing and generating FEN strings for chess positions.
"""

from typing import Tuple, Optional
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from core.board import ChessBoard, Piece, Color, CastlingRights


class FENParser:
    """Parser for FEN notation."""
    
    PIECE_SYMBOLS = {
        'P': (Piece.PAWN, Color.WHITE),
        'N': (Piece.KNIGHT, Color.WHITE),
        'B': (Piece.BISHOP, Color.WHITE),
        'R': (Piece.ROOK, Color.WHITE),
        'Q': (Piece.QUEEN, Color.WHITE),
        'K': (Piece.KING, Color.WHITE),
        'p': (Piece.PAWN, Color.BLACK),
        'n': (Piece.KNIGHT, Color.BLACK),
        'b': (Piece.BISHOP, Color.BLACK),
        'r': (Piece.ROOK, Color.BLACK),
        'q': (Piece.QUEEN, Color.BLACK),
        'k': (Piece.KING, Color.BLACK),
    }
    
    SYMBOL_TO_PIECE = {v: k for k, v in PIECE_SYMBOLS.items()}
    
    @classmethod
    def parse_fen(cls, fen: str) -> ChessBoard:
        """
        Parse FEN string and return ChessBoard.
        
        FEN format: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        """
        parts = fen.strip().split()
        if len(parts) != 6:
            raise ValueError(f"Invalid FEN: expected 6 parts, got {len(parts)}")
        
        piece_placement, active_color, castling, en_passant, halfmove, fullmove = parts
        
        board = ChessBoard()
        
        # Parse piece placement
        cls._parse_piece_placement(board, piece_placement)
        
        # Parse active color
        board.turn = Color.WHITE if active_color.lower() == 'w' else Color.BLACK
        
        # Parse castling rights
        board.castling_rights = cls._parse_castling_rights(castling)
        
        # Parse en passant target
        board.en_passant_target = cls._parse_en_passant(en_passant)
        
        # Parse halfmove clock
        try:
            board.halfmove_clock = int(halfmove)
        except ValueError:
            raise ValueError(f"Invalid halfmove clock: {halfmove}")
        
        # Parse fullmove number
        try:
            board.fullmove_number = int(fullmove)
        except ValueError:
            raise ValueError(f"Invalid fullmove number: {fullmove}")
        
        # Initialize zobrist hash
        board.zobrist_hash = board.zobrist_hasher.hash_position(board)
        
        return board
    
    @classmethod
    def _parse_piece_placement(cls, board: ChessBoard, placement: str):
        """Parse piece placement part of FEN."""
        ranks = placement.split('/')
        if len(ranks) != 8:
            raise ValueError(f"Invalid piece placement: expected 8 ranks, got {len(ranks)}")
        
        for rank_idx, rank in enumerate(ranks):
            file_idx = 0
            for char in rank:
                if char.isdigit():
                    # Empty squares
                    empty_count = int(char)
                    if empty_count < 1 or empty_count > 8:
                        raise ValueError(f"Invalid empty square count: {char}")
                    file_idx += empty_count
                elif char in cls.PIECE_SYMBOLS:
                    # Piece
                    if file_idx >= 8:
                        raise ValueError(f"Too many files in rank {rank}")
                    
                    piece, color = cls.PIECE_SYMBOLS[char]
                    square = (7 - rank_idx) * 8 + file_idx
                    board.set_piece_at(square, piece, color)
                    file_idx += 1
                else:
                    raise ValueError(f"Invalid character in piece placement: {char}")
            
            if file_idx != 8:
                raise ValueError(f"Incomplete rank {rank}: {file_idx} files")
    
    @classmethod
    def _parse_castling_rights(cls, castling: str) -> CastlingRights:
        """Parse castling rights from FEN."""
        if castling == '-':
            return CastlingRights(False, False, False, False)
        
        rights = CastlingRights()
        rights.white_kingside = 'K' in castling
        rights.white_queenside = 'Q' in castling
        rights.black_kingside = 'k' in castling
        rights.black_queenside = 'q' in castling
        
        return rights
    
    @classmethod
    def _parse_en_passant(cls, en_passant: str) -> Optional[int]:
        """Parse en passant target square from FEN."""
        if en_passant == '-':
            return None
        
        if len(en_passant) != 2:
            raise ValueError(f"Invalid en passant target: {en_passant}")
        
        try:
            file = ord(en_passant[0].lower()) - ord('a')
            rank = int(en_passant[1]) - 1
            
            if not (0 <= file <= 7) or not (0 <= rank <= 7):
                raise ValueError(f"Invalid en passant target: {en_passant}")
            
            return rank * 8 + file
        except (ValueError, IndexError):
            raise ValueError(f"Invalid en passant target: {en_passant}")
    
    @classmethod
    def to_fen(cls, board: ChessBoard) -> str:
        """Convert ChessBoard to FEN string."""
        # Piece placement
        placement_parts = []
        for rank in range(7, -1, -1):  # 8th rank to 1st rank
            rank_str = ""
            empty_count = 0
            
            for file in range(8):
                square = rank * 8 + file
                piece, color = board.get_piece_at(square)
                
                if piece == Piece.EMPTY:
                    empty_count += 1
                else:
                    if empty_count > 0:
                        rank_str += str(empty_count)
                        empty_count = 0
                    
                    symbol = cls.SYMBOL_TO_PIECE.get((piece, color))
                    if symbol:
                        rank_str += symbol
            
            if empty_count > 0:
                rank_str += str(empty_count)
            
            placement_parts.append(rank_str)
        
        piece_placement = "/".join(placement_parts)
        
        # Active color
        active_color = 'w' if board.turn == Color.WHITE else 'b'
        
        # Castling rights
        castling = ""
        if board.castling_rights.white_kingside:
            castling += "K"
        if board.castling_rights.white_queenside:
            castling += "Q"
        if board.castling_rights.black_kingside:
            castling += "k"
        if board.castling_rights.black_queenside:
            castling += "q"
        if not castling:
            castling = "-"
        
        # En passant target
        if board.en_passant_target is not None:
            file = chr(ord('a') + (board.en_passant_target % 8))
            rank = str((board.en_passant_target // 8) + 1)
            en_passant = file + rank
        else:
            en_passant = "-"
        
        # Halfmove clock and fullmove number
        halfmove = str(board.halfmove_clock)
        fullmove = str(board.fullmove_number)
        
        return f"{piece_placement} {active_color} {castling} {en_passant} {halfmove} {fullmove}"


# Convenience functions
def parse_fen(fen: str) -> ChessBoard:
    """Parse FEN string to ChessBoard."""
    return FENParser.parse_fen(fen)


def to_fen(board: ChessBoard) -> str:
    """Convert ChessBoard to FEN string."""
    return FENParser.to_fen(board)


# Standard starting position
STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
