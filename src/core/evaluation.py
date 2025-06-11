"""
Chess Position Evaluation
========================

Basic evaluation function for chess positions.
This will be replaced/enhanced with neural network evaluation in later phases.
"""

from typing import Dict, Tuple
import numpy as np
from .board import ChessBoard, Piece, Color, BitBoard


class Evaluator:
    """Chess position evaluator with basic heuristics."""
    
    def __init__(self):
        """Initialize evaluator with piece values and position tables."""
        # Basic piece values (centipawns)
        self.piece_values = {
            Piece.PAWN: 100,
            Piece.KNIGHT: 320,
            Piece.BISHOP: 330,
            Piece.ROOK: 500,
            Piece.QUEEN: 900,
            Piece.KING: 20000
        }
        
        # Piece-square tables for positional evaluation
        self._init_piece_square_tables()
    
    def _init_piece_square_tables(self):
        """Initialize piece-square tables for positional evaluation."""
        # Pawn piece-square table (from white's perspective)
        self.pawn_table = np.array([
            [ 0,  0,  0,  0,  0,  0,  0,  0],
            [50, 50, 50, 50, 50, 50, 50, 50],
            [10, 10, 20, 30, 30, 20, 10, 10],
            [ 5,  5, 10, 25, 25, 10,  5,  5],
            [ 0,  0,  0, 20, 20,  0,  0,  0],
            [ 5, -5,-10,  0,  0,-10, -5,  5],
            [ 5, 10, 10,-20,-20, 10, 10,  5],
            [ 0,  0,  0,  0,  0,  0,  0,  0]
        ])
        
        # Knight piece-square table
        self.knight_table = np.array([
            [-50,-40,-30,-30,-30,-30,-40,-50],
            [-40,-20,  0,  0,  0,  0,-20,-40],
            [-30,  0, 10, 15, 15, 10,  0,-30],
            [-30,  5, 15, 20, 20, 15,  5,-30],
            [-30,  0, 15, 20, 20, 15,  0,-30],
            [-30,  5, 10, 15, 15, 10,  5,-30],
            [-40,-20,  0,  5,  5,  0,-20,-40],
            [-50,-40,-30,-30,-30,-30,-40,-50]
        ])
        
        # Bishop piece-square table
        self.bishop_table = np.array([
            [-20,-10,-10,-10,-10,-10,-10,-20],
            [-10,  0,  0,  0,  0,  0,  0,-10],
            [-10,  0,  5, 10, 10,  5,  0,-10],
            [-10,  5,  5, 10, 10,  5,  5,-10],
            [-10,  0, 10, 10, 10, 10,  0,-10],
            [-10, 10, 10, 10, 10, 10, 10,-10],
            [-10,  5,  0,  0,  0,  0,  5,-10],
            [-20,-10,-10,-10,-10,-10,-10,-20]
        ])
        
        # Rook piece-square table
        self.rook_table = np.array([
            [ 0,  0,  0,  0,  0,  0,  0,  0],
            [ 5, 10, 10, 10, 10, 10, 10,  5],
            [-5,  0,  0,  0,  0,  0,  0, -5],
            [-5,  0,  0,  0,  0,  0,  0, -5],
            [-5,  0,  0,  0,  0,  0,  0, -5],
            [-5,  0,  0,  0,  0,  0,  0, -5],
            [-5,  0,  0,  0,  0,  0,  0, -5],
            [ 0,  0,  0,  5,  5,  0,  0,  0]
        ])
        
        # Queen piece-square table
        self.queen_table = np.array([
            [-20,-10,-10, -5, -5,-10,-10,-20],
            [-10,  0,  0,  0,  0,  0,  0,-10],
            [-10,  0,  5,  5,  5,  5,  0,-10],
            [ -5,  0,  5,  5,  5,  5,  0, -5],
            [  0,  0,  5,  5,  5,  5,  0, -5],
            [-10,  5,  5,  5,  5,  5,  0,-10],
            [-10,  0,  5,  0,  0,  0,  0,-10],
            [-20,-10,-10, -5, -5,-10,-10,-20]
        ])
        
        # King piece-square table (middlegame)
        self.king_table_mg = np.array([
            [-30,-40,-40,-50,-50,-40,-40,-30],
            [-30,-40,-40,-50,-50,-40,-40,-30],
            [-30,-40,-40,-50,-50,-40,-40,-30],
            [-30,-40,-40,-50,-50,-40,-40,-30],
            [-20,-30,-30,-40,-40,-30,-30,-20],
            [-10,-20,-20,-20,-20,-20,-20,-10],
            [ 20, 20,  0,  0,  0,  0, 20, 20],
            [ 20, 30, 10,  0,  0, 10, 30, 20]
        ])
        
        # King piece-square table (endgame)
        self.king_table_eg = np.array([
            [-50,-40,-30,-20,-20,-30,-40,-50],
            [-30,-20,-10,  0,  0,-10,-20,-30],
            [-30,-10, 20, 30, 30, 20,-10,-30],
            [-30,-10, 30, 40, 40, 30,-10,-30],
            [-30,-10, 30, 40, 40, 30,-10,-30],
            [-30,-10, 20, 30, 30, 20,-10,-30],
            [-30,-30,  0,  0,  0,  0,-30,-30],
            [-50,-30,-30,-30,-30,-30,-30,-50]
        ])
        
        # Store tables in a dictionary for easy access
        self.piece_square_tables = {
            Piece.PAWN: self.pawn_table,
            Piece.KNIGHT: self.knight_table,
            Piece.BISHOP: self.bishop_table,
            Piece.ROOK: self.rook_table,
            Piece.QUEEN: self.queen_table,
            Piece.KING: self.king_table_mg  # Will switch to endgame table when appropriate
        }
    
    def evaluate(self, board: ChessBoard) -> float:
        """
        Evaluate the current position from white's perspective.
        Returns a score in centipawns (positive = white advantage).
        """
        score = 0.0
        
        # Material and positional evaluation
        score += self._evaluate_material_and_position(board)
        
        # Additional positional factors
        score += self._evaluate_king_safety(board)
        score += self._evaluate_pawn_structure(board)
        score += self._evaluate_piece_mobility(board)
        score += self._evaluate_center_control(board)
        
        # Return score from the perspective of the player to move
        if board.to_move == Color.BLACK:
            score = -score
        
        return score
    
    def _evaluate_material_and_position(self, board: ChessBoard) -> float:
        """Evaluate material balance and piece positioning."""
        score = 0.0
        
        for color in [Color.WHITE, Color.BLACK]:
            color_multiplier = 1 if color == Color.WHITE else -1
            
            for piece in [Piece.PAWN, Piece.KNIGHT, Piece.BISHOP, 
                         Piece.ROOK, Piece.QUEEN, Piece.KING]:
                piece_bb = board.bitboards[(color, piece)]
                
                # Count pieces and add material value
                piece_count = piece_bb.pop_count()
                score += color_multiplier * piece_count * self.piece_values[piece]
                
                # Add positional bonus from piece-square tables
                current_bb = BitBoard(piece_bb.value)
                _, square = current_bb.pop_ls1b()
                while square != -1:
                    file, rank = square % 8, square // 8
                    
                    # Flip rank for black pieces (piece-square tables are from white's perspective)
                    table_rank = rank if color == Color.WHITE else 7 - rank
                    
                    if piece in self.piece_square_tables:
                        position_bonus = self.piece_square_tables[piece][table_rank][file]
                        score += color_multiplier * position_bonus
                    
                    current_bb, square = current_bb.pop_ls1b()
        
        return score
    
    def _evaluate_king_safety(self, board: ChessBoard) -> float:
        """Evaluate king safety (basic implementation)."""
        score = 0.0
        
        for color in [Color.WHITE, Color.BLACK]:
            color_multiplier = 1 if color == Color.WHITE else -1
            king_bb = board.bitboards[(color, Piece.KING)]
            
            if king_bb:
                _, king_square = king_bb.pop_ls1b()
                file, rank = king_square % 8, king_square // 8
                
                # Penalty for king in center during opening/middlegame
                if self._is_opening_or_middlegame(board):
                    if 2 <= file <= 5 and 2 <= rank <= 5:
                        score += color_multiplier * -50  # Center king penalty
                
                # Bonus for castled king (king on g or c file with rook nearby)
                if color == Color.WHITE and rank == 0:
                    if file == 6 or file == 2:  # Likely castled position
                        score += color_multiplier * 50
                elif color == Color.BLACK and rank == 7:
                    if file == 6 or file == 2:  # Likely castled position
                        score += color_multiplier * 50
        
        return score
    
    def _evaluate_pawn_structure(self, board: ChessBoard) -> float:
        """Evaluate pawn structure quality."""
        score = 0.0
        
        for color in [Color.WHITE, Color.BLACK]:
            color_multiplier = 1 if color == Color.WHITE else -1
            pawn_bb = board.bitboards[(color, Piece.PAWN)]
            
            # Count pawns on each file
            file_counts = [0] * 8
            current_bb = BitBoard(pawn_bb.value)
            _, square = current_bb.pop_ls1b()
            while square != -1:
                file = square % 8
                file_counts[file] += 1
                current_bb, square = current_bb.pop_ls1b()
            
            # Doubled pawns penalty
            for file in range(8):
                if file_counts[file] > 1:
                    score += color_multiplier * -50 * (file_counts[file] - 1)
            
            # Isolated pawns penalty
            for file in range(8):
                if file_counts[file] > 0:
                    isolated = True
                    if file > 0 and file_counts[file - 1] > 0:
                        isolated = False
                    if file < 7 and file_counts[file + 1] > 0:
                        isolated = False
                    
                    if isolated:
                        score += color_multiplier * -20
        
        return score
    
    def _evaluate_piece_mobility(self, board: ChessBoard) -> float:
        """Evaluate piece mobility (simplified)."""
        score = 0.0
        
        # Import here to avoid circular imports
        from .moves import MoveGenerator
        
        move_gen = MoveGenerator()
        
        # Count legal moves for current player
        current_moves = len(move_gen.generate_moves(board))
        
        # Count legal moves for opponent (switch sides temporarily)
        board_copy = board.copy()
        board_copy.to_move = Color.BLACK if board.to_move == Color.WHITE else Color.WHITE
        opponent_moves = len(move_gen.generate_moves(board_copy))
        
        # Bonus for having more mobility
        mobility_bonus = (current_moves - opponent_moves) * 2
        score += mobility_bonus if board.to_move == Color.WHITE else -mobility_bonus
        
        return score
    
    def _evaluate_center_control(self, board: ChessBoard) -> float:
        """Evaluate control of central squares."""
        score = 0.0
        
        # Central squares: e4, e5, d4, d5
        central_squares = [28, 29, 35, 36]  # e4, e5, d4, d5 in 0-63 notation
        
        for color in [Color.WHITE, Color.BLACK]:
            color_multiplier = 1 if color == Color.WHITE else -1
            
            # Check for pawns on central squares
            pawn_bb = board.bitboards[(color, Piece.PAWN)]
            for square in central_squares:
                if pawn_bb.get_bit(square):
                    score += color_multiplier * 30
            
            # Check for pieces attacking central squares
            # (Simplified - just check piece presence near center)
            for piece in [Piece.KNIGHT, Piece.BISHOP]:
                piece_bb = board.bitboards[(color, piece)]
                current_bb = BitBoard(piece_bb.value)
                _, square = current_bb.pop_ls1b()
                while square != -1:
                    file, rank = square % 8, square // 8
                    # Bonus for pieces near center
                    center_distance = abs(file - 3.5) + abs(rank - 3.5)
                    if center_distance < 3:
                        score += color_multiplier * (10 - int(center_distance * 2))
                    current_bb, square = current_bb.pop_ls1b()
        
        return score
    
    def _is_opening_or_middlegame(self, board: ChessBoard) -> bool:
        """Determine if we're in opening or middlegame phase."""
        # Simple heuristic: count major pieces
        major_pieces = 0
        for color in [Color.WHITE, Color.BLACK]:
            for piece in [Piece.QUEEN, Piece.ROOK, Piece.BISHOP, Piece.KNIGHT]:
                major_pieces += board.bitboards[(color, piece)].pop_count()
        
        # If we have most major pieces, we're likely in opening/middlegame
        return major_pieces >= 16  # Starting position has 20 major pieces
    
    def _is_endgame(self, board: ChessBoard) -> bool:
        """Determine if we're in endgame phase."""
        return not self._is_opening_or_middlegame(board)
    
    def get_material_balance(self, board: ChessBoard) -> float:
        """Get pure material balance."""
        balance = 0.0
        
        for color in [Color.WHITE, Color.BLACK]:
            color_multiplier = 1 if color == Color.WHITE else -1
            
            for piece in [Piece.PAWN, Piece.KNIGHT, Piece.BISHOP, 
                         Piece.ROOK, Piece.QUEEN]:
                piece_count = board.bitboards[(color, piece)].pop_count()
                balance += color_multiplier * piece_count * self.piece_values[piece]
        
        return balance
    
    def is_draw_by_insufficient_material(self, board: ChessBoard) -> bool:
        """Check if position is drawn by insufficient material."""
        # Count material for both sides
        white_pieces = []
        black_pieces = []
        
        for piece in [Piece.PAWN, Piece.KNIGHT, Piece.BISHOP, Piece.ROOK, Piece.QUEEN]:
            white_count = board.bitboards[(Color.WHITE, piece)].pop_count()
            black_count = board.bitboards[(Color.BLACK, piece)].pop_count()
            
            white_pieces.extend([piece] * white_count)
            black_pieces.extend([piece] * black_count)
        
        # Remove kings
        total_pieces = len(white_pieces) + len(black_pieces)
        
        # King vs King
        if total_pieces == 0:
            return True
        
        # King + minor piece vs King
        if total_pieces == 1:
            piece = white_pieces[0] if white_pieces else black_pieces[0]
            if piece in [Piece.BISHOP, Piece.KNIGHT]:
                return True
        
        # King + Bishop vs King + Bishop (same color squares)
        if (total_pieces == 2 and 
            len(white_pieces) == 1 and len(black_pieces) == 1 and
            white_pieces[0] == Piece.BISHOP and black_pieces[0] == Piece.BISHOP):
            # Would need to check if bishops are on same color squares
            return True
        
        return False
