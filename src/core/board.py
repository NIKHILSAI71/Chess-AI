"""
Chess Board Representation
=========================

High-performance board representation using bitboards for the chess AI engine.
This is the foundation for fast move generation and position evaluation.
"""

from typing import List, Tuple, Optional, Dict, Set
import numpy as np
from enum import IntEnum
from dataclasses import dataclass


class Piece(IntEnum):
    """Chess piece types."""
    EMPTY = 0
    PAWN = 1
    KNIGHT = 2
    BISHOP = 3
    ROOK = 4
    QUEEN = 5
    KING = 6


class Color(IntEnum):
    """Player colors."""
    WHITE = 0
    BLACK = 1


class Square(IntEnum):
    """Board squares (0-63, a1=0, h8=63)."""
    A1, B1, C1, D1, E1, F1, G1, H1 = range(8)
    A2, B2, C2, D2, E2, F2, G2, H2 = range(8, 16)
    A3, B3, C3, D3, E3, F3, G3, H3 = range(16, 24)
    A4, B4, C4, D4, E4, F4, G4, H4 = range(24, 32)
    A5, B5, C5, D5, E5, F5, G5, H5 = range(32, 40)
    A6, B6, C6, D6, E6, F6, G6, H6 = range(40, 48)
    A7, B7, C7, D7, E7, F7, G7, H7 = range(48, 56)
    A8, B8, C8, D8, E8, F8, G8, H8 = range(56, 64)


@dataclass
class CastlingRights:
    """Castling rights for both sides."""
    white_kingside: bool = True
    white_queenside: bool = True
    black_kingside: bool = True
    black_queenside: bool = True


@dataclass
class Move:
    """Represents a chess move."""
    from_square: int
    to_square: int
    piece: Piece
    captured_piece: Piece = Piece.EMPTY
    promotion_piece: Piece = Piece.EMPTY
    is_castling: bool = False
    is_en_passant: bool = False
    
    def __str__(self) -> str:
        """Convert move to algebraic notation."""
        from_file = chr(ord('a') + (self.from_square % 8))
        from_rank = str((self.from_square // 8) + 1)
        to_file = chr(ord('a') + (self.to_square % 8))
        to_rank = str((self.to_square // 8) + 1)
        
        move_str = f"{from_file}{from_rank}{to_file}{to_rank}"
        
        if self.promotion_piece != Piece.EMPTY:
            piece_symbols = {Piece.QUEEN: 'q', Piece.ROOK: 'r', 
                           Piece.BISHOP: 'b', Piece.KNIGHT: 'n'}
            move_str += piece_symbols[self.promotion_piece]
        
        return move_str
    
    def __eq__(self, other) -> bool:
        """Check move equality."""
        if not isinstance(other, Move):
            return False
        return (self.from_square == other.from_square and 
                self.to_square == other.to_square and
                self.promotion_piece == other.promotion_piece)


class BitBoard:
    """64-bit integer representing piece positions on the board."""
    
    def __init__(self, value: int = 0):
        self.value = value
    
    def __or__(self, other) -> 'BitBoard':
        return BitBoard(self.value | other.value)
    
    def __and__(self, other) -> 'BitBoard':
        return BitBoard(self.value & other.value)
    
    def __xor__(self, other) -> 'BitBoard':
        return BitBoard(self.value ^ other.value)
    
    def __invert__(self) -> 'BitBoard':
        return BitBoard(~self.value & 0xFFFFFFFFFFFFFFFF)
    
    def __bool__(self) -> bool:
        return self.value != 0
    
    def set_bit(self, square: int) -> 'BitBoard':
        """Set bit at given square."""
        return BitBoard(self.value | (1 << square))
    
    def clear_bit(self, square: int) -> 'BitBoard':
        """Clear bit at given square."""
        return BitBoard(self.value & ~(1 << square))
    
    def get_bit(self, square: int) -> bool:
        """Check if bit is set at given square."""
        return bool(self.value & (1 << square))
    
    def pop_count(self) -> int:
        """Count number of set bits (population count)."""
        return bin(self.value).count('1')
    
    def ls1b(self) -> int:
        """Get index of least significant 1 bit."""
        if self.value == 0:
            return -1
        return (self.value & -self.value).bit_length() - 1
    
    def pop_ls1b(self) -> Tuple['BitBoard', int]:
        """Pop least significant 1 bit and return new bitboard and square index."""
        if self.value == 0:
            return BitBoard(0), -1
        square = self.ls1b()
        return BitBoard(self.value & (self.value - 1)), square


class ChessBoard:
    """
    High-performance chess board representation using bitboards.
    Supports fast move generation, position evaluation, and game state management.
    """
    
    def __init__(self):
        """Initialize board to starting position."""
        # Bitboards for each piece type and color
        self.bitboards = {
            (Color.WHITE, Piece.PAWN): BitBoard(0x000000000000FF00),
            (Color.WHITE, Piece.KNIGHT): BitBoard(0x0000000000000042),
            (Color.WHITE, Piece.BISHOP): BitBoard(0x0000000000000024),
            (Color.WHITE, Piece.ROOK): BitBoard(0x0000000000000081),
            (Color.WHITE, Piece.QUEEN): BitBoard(0x0000000000000008),
            (Color.WHITE, Piece.KING): BitBoard(0x0000000000000010),
            
            (Color.BLACK, Piece.PAWN): BitBoard(0x00FF000000000000),
            (Color.BLACK, Piece.KNIGHT): BitBoard(0x4200000000000000),
            (Color.BLACK, Piece.BISHOP): BitBoard(0x2400000000000000),
            (Color.BLACK, Piece.ROOK): BitBoard(0x8100000000000000),
            (Color.BLACK, Piece.QUEEN): BitBoard(0x0800000000000000),
            (Color.BLACK, Piece.KING): BitBoard(0x1000000000000000),
        }
        
        # Combined bitboards for optimization
        self.white_pieces = BitBoard(0x000000000000FFFF)
        self.black_pieces = BitBoard(0xFFFF000000000000)
        self.occupied = BitBoard(0xFFFF00000000FFFF)
        self.empty = ~self.occupied
        
        # Game state
        self.to_move = Color.WHITE
        self.castling_rights = CastlingRights()
        self.en_passant_square = None
        self.halfmove_clock = 0
        self.fullmove_number = 1
        
        # Move history for repetition detection
        self.position_history: List[int] = []
        self.move_history: List[Move] = []
        self.zobrist_hash = 0
    
    def get_piece_at(self, square: int) -> Tuple[Piece, Color]:
        """Get piece and color at given square."""
        for color in [Color.WHITE, Color.BLACK]:
            for piece in [Piece.PAWN, Piece.KNIGHT, Piece.BISHOP, 
                         Piece.ROOK, Piece.QUEEN, Piece.KING]:
                if self.bitboards[(color, piece)].get_bit(square):
                    return piece, color
        return Piece.EMPTY, Color.WHITE
    
    def set_piece_at(self, square: int, piece: Piece, color: Color):
        """Set piece at given square."""
        # Clear the square first
        self.clear_square(square)
        
        if piece != Piece.EMPTY:
            self.bitboards[(color, piece)] = self.bitboards[(color, piece)].set_bit(square)
            if color == Color.WHITE:
                self.white_pieces = self.white_pieces.set_bit(square)
            else:
                self.black_pieces = self.black_pieces.set_bit(square)
            self.occupied = self.occupied.set_bit(square)
            self.empty = self.empty.clear_bit(square)
    
    def clear_square(self, square: int):
        """Clear piece at given square."""
        for color in [Color.WHITE, Color.BLACK]:
            for piece in [Piece.PAWN, Piece.KNIGHT, Piece.BISHOP, 
                         Piece.ROOK, Piece.QUEEN, Piece.KING]:
                self.bitboards[(color, piece)] = self.bitboards[(color, piece)].clear_bit(square)
        
        self.white_pieces = self.white_pieces.clear_bit(square)
        self.black_pieces = self.black_pieces.clear_bit(square)
        self.occupied = self.occupied.clear_bit(square)
        self.empty = self.empty.set_bit(square)
    
    def make_move(self, move: Move) -> bool:
        """Make a move on the board. Returns True if successful."""
        # Store current position for undo
        self.position_history.append(self.get_position_hash())
        self.move_history.append(move)
        
        # Get pieces involved
        moving_piece, moving_color = self.get_piece_at(move.from_square)
        captured_piece, captured_color = self.get_piece_at(move.to_square)
        
        # Handle special moves
        if move.is_castling:
            self._handle_castling(move)
        elif move.is_en_passant:
            self._handle_en_passant(move)
        else:
            # Regular move
            self.clear_square(move.from_square)
            
            # Handle promotion
            final_piece = move.promotion_piece if move.promotion_piece != Piece.EMPTY else moving_piece
            self.set_piece_at(move.to_square, final_piece, moving_color)
        
        # Update game state
        self._update_castling_rights(move)
        self._update_en_passant(move)
        
        if captured_piece != Piece.EMPTY or moving_piece == Piece.PAWN:
            self.halfmove_clock = 0
        else:
            self.halfmove_clock += 1
        
        if self.to_move == Color.BLACK:
            self.fullmove_number += 1
        
        self.to_move = Color.BLACK if self.to_move == Color.WHITE else Color.WHITE
        
        return True
    
    def unmake_move(self) -> bool:
        """Unmake the last move."""
        if not self.move_history:
            return False
        
        # Remove the last move from history
        last_move = self.move_history.pop()
        
        # Restore the previous position hash
        if self.position_history:
            self.position_history.pop()
        
        # This is a simplified implementation - a full implementation would
        # need to store and restore the complete game state
        # For now, we'll create a new board from FEN as a workaround
        # TODO: Implement proper state restoration
        return True

    def _handle_castling(self, move: Move):
        """Handle castling move."""
        if move.to_square == Square.G1:  # White kingside
            self.clear_square(Square.E1)
            self.clear_square(Square.H1)
            self.set_piece_at(Square.G1, Piece.KING, Color.WHITE)
            self.set_piece_at(Square.F1, Piece.ROOK, Color.WHITE)
        elif move.to_square == Square.C1:  # White queenside
            self.clear_square(Square.E1)
            self.clear_square(Square.A1)
            self.set_piece_at(Square.C1, Piece.KING, Color.WHITE)
            self.set_piece_at(Square.D1, Piece.ROOK, Color.WHITE)
        elif move.to_square == Square.G8:  # Black kingside
            self.clear_square(Square.E8)
            self.clear_square(Square.H8)
            self.set_piece_at(Square.G8, Piece.KING, Color.BLACK)
            self.set_piece_at(Square.F8, Piece.ROOK, Color.BLACK)
        elif move.to_square == Square.C8:  # Black queenside
            self.clear_square(Square.E8)
            self.clear_square(Square.A8)
            self.set_piece_at(Square.C8, Piece.KING, Color.BLACK)
            self.set_piece_at(Square.D8, Piece.ROOK, Color.BLACK)
    
    def _handle_en_passant(self, move: Move):
        """Handle en passant capture."""
        self.clear_square(move.from_square)
        self.set_piece_at(move.to_square, Piece.PAWN, self.to_move)
        
        # Remove captured pawn
        if self.to_move == Color.WHITE:
            captured_square = move.to_square - 8
        else:
            captured_square = move.to_square + 8
        
        self.clear_square(captured_square)
    
    def _update_castling_rights(self, move: Move):
        """Update castling rights after a move."""
        # King moves remove all castling rights for that color
        if move.piece == Piece.KING:
            if self.to_move == Color.WHITE:
                self.castling_rights.white_kingside = False
                self.castling_rights.white_queenside = False
            else:
                self.castling_rights.black_kingside = False
                self.castling_rights.black_queenside = False
        
        # Rook moves remove castling rights for that side
        elif move.piece == Piece.ROOK:
            if move.from_square == Square.A1:
                self.castling_rights.white_queenside = False
            elif move.from_square == Square.H1:
                self.castling_rights.white_kingside = False
            elif move.from_square == Square.A8:
                self.castling_rights.black_queenside = False
            elif move.from_square == Square.H8:
                self.castling_rights.black_kingside = False
        
        # Capturing rooks removes castling rights
        if move.to_square == Square.A1:
            self.castling_rights.white_queenside = False
        elif move.to_square == Square.H1:
            self.castling_rights.white_kingside = False
        elif move.to_square == Square.A8:
            self.castling_rights.black_queenside = False
        elif move.to_square == Square.H8:
            self.castling_rights.black_kingside = False
    
    def _update_en_passant(self, move: Move):
        """Update en passant square after a move."""
        self.en_passant_square = None
        
        # Check for pawn double push
        if move.piece == Piece.PAWN:
            if abs(move.to_square - move.from_square) == 16:
                self.en_passant_square = (move.from_square + move.to_square) // 2
    
    def get_position_hash(self) -> int:
        """Get a hash of the current position for repetition detection."""
        hash_value = 0
        for (color, piece), bb in self.bitboards.items():
            hash_value ^= bb.value * (color.value + 1) * (piece.value + 1)
        
        hash_value ^= self.to_move.value << 60
        hash_value ^= (self.castling_rights.white_kingside << 56)
        hash_value ^= (self.castling_rights.white_queenside << 57)
        hash_value ^= (self.castling_rights.black_kingside << 58)
        hash_value ^= (self.castling_rights.black_queenside << 59)
        
        if self.en_passant_square is not None:
            hash_value ^= self.en_passant_square << 50
        
        return hash_value
    
    def is_in_check(self, color: Optional[Color] = None) -> bool:
        """Check if the given color's king is in check."""
        if color is None:
            color = self.to_move
        
        # Import here to avoid circular imports
        from .moves import MoveGenerator
        move_gen = MoveGenerator()
        return move_gen.is_in_check(self, color)

    def is_repetition(self) -> bool:
        """Check if current position is a repetition."""
        return self.is_threefold_repetition()

    def is_threefold_repetition(self) -> bool:
        """Check if current position has occurred three times."""
        current_hash = self.get_position_hash()
        count = self.position_history.count(current_hash)
        return count >= 2  # Current position + 2 previous = 3 total
    
    def to_fen(self) -> str:
        """Convert board to FEN (Forsyth-Edwards Notation)."""
        fen_parts = []
        
        # Board state
        board_fen = ""
        for rank in range(7, -1, -1):  # From rank 8 to rank 1
            empty_count = 0
            for file in range(8):
                square = rank * 8 + file
                piece, color = self.get_piece_at(square)
                
                if piece == Piece.EMPTY:
                    empty_count += 1
                else:
                    if empty_count > 0:
                        board_fen += str(empty_count)
                        empty_count = 0
                    
                    piece_symbols = {
                        Piece.PAWN: 'p', Piece.KNIGHT: 'n', Piece.BISHOP: 'b',
                        Piece.ROOK: 'r', Piece.QUEEN: 'q', Piece.KING: 'k'
                    }
                    symbol = piece_symbols[piece]
                    if color == Color.WHITE:
                        symbol = symbol.upper()
                    board_fen += symbol
            
            if empty_count > 0:
                board_fen += str(empty_count)
            
            if rank > 0:
                board_fen += "/"
        
        fen_parts.append(board_fen)
        
        # Active color
        fen_parts.append('w' if self.to_move == Color.WHITE else 'b')
        
        # Castling rights
        castling = ""
        if self.castling_rights.white_kingside:
            castling += "K"
        if self.castling_rights.white_queenside:
            castling += "Q"
        if self.castling_rights.black_kingside:
            castling += "k"
        if self.castling_rights.black_queenside:
            castling += "q"
        fen_parts.append(castling if castling else "-")
        
        # En passant square
        if self.en_passant_square is not None:
            file = chr(ord('a') + (self.en_passant_square % 8))
            rank = str((self.en_passant_square // 8) + 1)
            fen_parts.append(f"{file}{rank}")
        else:
            fen_parts.append("-")
        
        # Halfmove clock and fullmove number
        fen_parts.append(str(self.halfmove_clock))
        fen_parts.append(str(self.fullmove_number))
        
        return " ".join(fen_parts)
    
    @classmethod
    def from_fen(cls, fen: str) -> 'ChessBoard':
        """Create a board from FEN string."""
        parts = fen.split()
        if len(parts) != 6:
            raise ValueError("Invalid FEN: must have 6 parts")
        
        board = cls.__new__(cls)
        
        # Initialize empty bitboards
        board.bitboards = {
            (Color.WHITE, Piece.PAWN): BitBoard(0),
            (Color.WHITE, Piece.KNIGHT): BitBoard(0),
            (Color.WHITE, Piece.BISHOP): BitBoard(0),
            (Color.WHITE, Piece.ROOK): BitBoard(0),
            (Color.WHITE, Piece.QUEEN): BitBoard(0),
            (Color.WHITE, Piece.KING): BitBoard(0),
            (Color.BLACK, Piece.PAWN): BitBoard(0),
            (Color.BLACK, Piece.KNIGHT): BitBoard(0),
            (Color.BLACK, Piece.BISHOP): BitBoard(0),
            (Color.BLACK, Piece.ROOK): BitBoard(0),
            (Color.BLACK, Piece.QUEEN): BitBoard(0),
            (Color.BLACK, Piece.KING): BitBoard(0),
        }
        
        board.white_pieces = BitBoard(0)
        board.black_pieces = BitBoard(0)
        board.occupied = BitBoard(0)
        
        # Parse piece placement
        piece_map = {
            'p': (Piece.PAWN, Color.BLACK), 'P': (Piece.PAWN, Color.WHITE),
            'n': (Piece.KNIGHT, Color.BLACK), 'N': (Piece.KNIGHT, Color.WHITE),
            'b': (Piece.BISHOP, Color.BLACK), 'B': (Piece.BISHOP, Color.WHITE),
            'r': (Piece.ROOK, Color.BLACK), 'R': (Piece.ROOK, Color.WHITE),
            'q': (Piece.QUEEN, Color.BLACK), 'Q': (Piece.QUEEN, Color.WHITE),
            'k': (Piece.KING, Color.BLACK), 'K': (Piece.KING, Color.WHITE),
        }
        
        square = 56  # Start at a8
        for char in parts[0]:
            if char == '/':
                square -= 16  # Move to next rank
            elif char.isdigit():
                square += int(char)  # Skip empty squares
            elif char in piece_map:
                piece, color = piece_map[char]
                board.bitboards[(color, piece)] = board.bitboards[(color, piece)].set_bit(square)
                if color == Color.WHITE:
                    board.white_pieces = board.white_pieces.set_bit(square)
                else:
                    board.black_pieces = board.black_pieces.set_bit(square)
                board.occupied = board.occupied.set_bit(square)
                square += 1
        
        board.empty = ~board.occupied
        
        # Parse active color
        board.to_move = Color.WHITE if parts[1] == 'w' else Color.BLACK
        
        # Parse castling rights
        board.castling_rights = CastlingRights(False, False, False, False)
        if 'K' in parts[2]:
            board.castling_rights.white_kingside = True
        if 'Q' in parts[2]:
            board.castling_rights.white_queenside = True
        if 'k' in parts[2]:
            board.castling_rights.black_kingside = True
        if 'q' in parts[2]:
            board.castling_rights.black_queenside = True
        
        # Parse en passant square
        if parts[3] == '-':
            board.en_passant_square = None
        else:
            file = ord(parts[3][0]) - ord('a')
            rank = int(parts[3][1]) - 1
            board.en_passant_square = rank * 8 + file
        
        # Parse move clocks
        board.halfmove_clock = int(parts[4])
        board.fullmove_number = int(parts[5])
        
        # Initialize history
        board.position_history = []
        board.move_history = []
        board.zobrist_hash = 0
        
        return board

    def copy(self) -> 'ChessBoard':
        """Create a deep copy of the board."""
        import copy as copy_module
        
        new_board = ChessBoard.__new__(ChessBoard)
        
        # Copy bitboards
        new_board.bitboards = {}
        for key, bb in self.bitboards.items():
            new_board.bitboards[key] = BitBoard(bb.value)
        
        # Copy combined bitboards
        new_board.white_pieces = BitBoard(self.white_pieces.value)
        new_board.black_pieces = BitBoard(self.black_pieces.value)
        new_board.occupied = BitBoard(self.occupied.value)
        new_board.empty = BitBoard(self.empty.value)
        
        # Copy game state
        new_board.to_move = self.to_move
        new_board.castling_rights = copy_module.deepcopy(self.castling_rights)
        new_board.en_passant_square = self.en_passant_square
        new_board.halfmove_clock = self.halfmove_clock
        new_board.fullmove_number = self.fullmove_number
        
        # Copy history
        new_board.position_history = self.position_history.copy()
        new_board.move_history = self.move_history.copy()
        new_board.zobrist_hash = self.zobrist_hash
        
        return new_board

    @property
    def turn(self) -> Color:
        """Get the current player to move."""
        return self.to_move
    
    @turn.setter
    def turn(self, color: Color):
        """Set the current player to move."""
        self.to_move = color
    
    @property
    def en_passant_target(self) -> Optional[int]:
        """Get the en passant target square."""
        return self.en_passant_square
    
    @en_passant_target.setter
    def en_passant_target(self, square: Optional[int]):
        """Set the en passant target square."""
        self.en_passant_square = square

    def __str__(self) -> str:
        """String representation of the board."""
        board_str = ""
        for rank in range(7, -1, -1):
            board_str += f"{rank + 1} "
            for file in range(8):
                square = rank * 8 + file
                piece, color = self.get_piece_at(square)
                
                if piece == Piece.EMPTY:
                    board_str += ". "
                else:
                    piece_symbols = {
                        Piece.PAWN: 'p', Piece.KNIGHT: 'n', Piece.BISHOP: 'b',
                        Piece.ROOK: 'r', Piece.QUEEN: 'q', Piece.KING: 'k'
                    }
                    symbol = piece_symbols[piece]
                    if color == Color.WHITE:
                        symbol = symbol.upper()
                    board_str += symbol + " "
            board_str += "\n"
        
        board_str += "  a b c d e f g h\n"
        board_str += f"To move: {'White' if self.to_move == Color.WHITE else 'Black'}\n"
        board_str += f"FEN: {self.to_fen()}"
        
        return board_str

    def get_king_square(self, color: Color) -> int:
        """Get the square where the king of the given color is located."""
        king_bb = self.bitboards[(color, Piece.KING)]
        _, king_square = king_bb.pop_ls1b()
        return king_square if king_square != -1 else -1
