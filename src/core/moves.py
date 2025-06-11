"""
Chess Move Generation Engine
===========================

High-performance move generation using bitboard operations.
Generates all legal moves for a given position efficiently.
"""

from typing import List, Iterator, Tuple
from .board import ChessBoard, Move, Piece, Color, Square, BitBoard


class AttackTables:
    """Pre-computed attack tables for pieces."""
    
    def __init__(self):
        """Initialize attack tables."""
        self.knight_attacks = [BitBoard(0) for _ in range(64)]
        self.king_attacks = [BitBoard(0) for _ in range(64)]
        self.pawn_attacks = {Color.WHITE: [BitBoard(0) for _ in range(64)],
                           Color.BLACK: [BitBoard(0) for _ in range(64)]}
        
        # Pre-compute attack patterns
        self._init_knight_attacks()
        self._init_king_attacks()
        self._init_pawn_attacks()
        
        # Sliding piece attacks (computed dynamically with magic bitboards could be added)
        self.bishop_masks = [BitBoard(0) for _ in range(64)]
        self.rook_masks = [BitBoard(0) for _ in range(64)]
        self._init_sliding_masks()
    
    def _init_knight_attacks(self):
        """Initialize knight attack patterns."""
        knight_moves = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                       (1, -2), (1, 2), (2, -1), (2, 1)]
        
        for square in range(64):
            file, rank = square % 8, square // 8
            attacks = BitBoard(0)
            
            for df, dr in knight_moves:
                new_file, new_rank = file + df, rank + dr
                if 0 <= new_file < 8 and 0 <= new_rank < 8:
                    new_square = new_rank * 8 + new_file
                    attacks = attacks.set_bit(new_square)
            
            self.knight_attacks[square] = attacks
    
    def _init_king_attacks(self):
        """Initialize king attack patterns."""
        king_moves = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                     (0, 1), (1, -1), (1, 0), (1, 1)]
        
        for square in range(64):
            file, rank = square % 8, square // 8
            attacks = BitBoard(0)
            
            for df, dr in king_moves:
                new_file, new_rank = file + df, rank + dr
                if 0 <= new_file < 8 and 0 <= new_rank < 8:
                    new_square = new_rank * 8 + new_file
                    attacks = attacks.set_bit(new_square)
            
            self.king_attacks[square] = attacks
    
    def _init_pawn_attacks(self):
        """Initialize pawn attack patterns."""
        for square in range(64):
            file, rank = square % 8, square // 8
            
            # White pawn attacks
            white_attacks = BitBoard(0)
            if rank < 7:  # Not on 8th rank
                if file > 0:  # Can capture left
                    white_attacks = white_attacks.set_bit((rank + 1) * 8 + file - 1)
                if file < 7:  # Can capture right
                    white_attacks = white_attacks.set_bit((rank + 1) * 8 + file + 1)
            self.pawn_attacks[Color.WHITE][square] = white_attacks
            
            # Black pawn attacks
            black_attacks = BitBoard(0)
            if rank > 0:  # Not on 1st rank
                if file > 0:  # Can capture left
                    black_attacks = black_attacks.set_bit((rank - 1) * 8 + file - 1)
                if file < 7:  # Can capture right
                    black_attacks = black_attacks.set_bit((rank - 1) * 8 + file + 1)
            self.pawn_attacks[Color.BLACK][square] = black_attacks
    
    def _init_sliding_masks(self):
        """Initialize sliding piece attack masks."""
        for square in range(64):
            file, rank = square % 8, square // 8
            
            # Bishop mask (diagonals)
            bishop_mask = BitBoard(0)
            # Positive diagonal
            for i in range(1, 8):
                f, r = file + i, rank + i
                if f < 7 and r < 7:  # Exclude edges for mask
                    bishop_mask = bishop_mask.set_bit(r * 8 + f)
                f, r = file - i, rank - i
                if f > 0 and r > 0:  # Exclude edges for mask
                    bishop_mask = bishop_mask.set_bit(r * 8 + f)
            # Negative diagonal
            for i in range(1, 8):
                f, r = file + i, rank - i
                if f < 7 and r > 0:  # Exclude edges for mask
                    bishop_mask = bishop_mask.set_bit(r * 8 + f)
                f, r = file - i, rank + i
                if f > 0 and r < 7:  # Exclude edges for mask
                    bishop_mask = bishop_mask.set_bit(r * 8 + f)
            self.bishop_masks[square] = bishop_mask
            
            # Rook mask (ranks and files)
            rook_mask = BitBoard(0)
            # Horizontal
            for f in range(1, 7):  # Exclude edges
                if f != file:
                    rook_mask = rook_mask.set_bit(rank * 8 + f)
            # Vertical
            for r in range(1, 7):  # Exclude edges
                if r != rank:
                    rook_mask = rook_mask.set_bit(r * 8 + file)
            self.rook_masks[square] = rook_mask


class MoveGenerator:
    """High-performance chess move generator."""
    
    def __init__(self):
        """Initialize move generator with pre-computed tables."""
        self.attack_tables = AttackTables()
    
    def generate_captures(self, board: ChessBoard) -> List[Move]:
        """Generate all capture moves for the current position."""
        return self.generate_moves(board, only_captures=True)
    def generate_legal_moves(self, board: ChessBoard) -> List[Move]:
        """Generate all legal moves for the current position."""
        return self.generate_moves(board, only_captures=False)
    
    def generate_pawn_moves(self, board: ChessBoard, color: Color) -> List[Move]:
        """Generate pawn moves for the given color."""
        return self._generate_pawn_moves(board, color, only_captures=False)
    
    def generate_knight_moves(self, board: ChessBoard, color: Color) -> List[Move]:
        """Generate knight moves for the given color."""
        return self._generate_knight_moves(board, color, only_captures=False)
    
    def generate_moves(self, board: ChessBoard, only_captures: bool = False) -> List[Move]:
        """Generate all legal moves for the current position."""
        moves = []
        color = board.to_move
        
        # Pre-compute important data for legal move generation
        opponent_color = Color.BLACK if color == Color.WHITE else Color.WHITE
        king_square = board.get_king_square(color)
        
        # Check if king is in check
        in_check = self._is_square_attacked(board, king_square, opponent_color)
        
        # Generate moves for each piece type - use a single list to avoid extend overhead
        self._generate_pawn_moves_fast(board, color, only_captures, moves)
        self._generate_knight_moves_fast(board, color, only_captures, moves)
        self._generate_sliding_moves_fast(board, color, only_captures, moves)
        self._generate_king_moves_fast(board, color, only_captures, moves)
        
        if not only_captures:
            self._generate_castling_moves_fast(board, color, moves)
        
        # Optimized legal move filtering
        if in_check:
            # When in check, we need stricter filtering
            legal_moves = []
            for move in moves:
                if self._is_legal_move_when_in_check(board, move, king_square, opponent_color):
                    legal_moves.append(move)
        else:
            # When not in check, use faster pin detection
            pinned_pieces = self._get_pinned_pieces(board, king_square, opponent_color)
            legal_moves = []
            for move in moves:
                if self._is_legal_move_optimized(board, move, king_square, pinned_pieces):
                    legal_moves.append(move)
        
        return legal_moves
    
    def _generate_pawn_moves(self, board: ChessBoard, color: Color, only_captures: bool) -> List[Move]:
        """Generate pawn moves."""
        moves = []
        pawn_bb = board.bitboards[(color, Piece.PAWN)]
        
        if color == Color.WHITE:
            forward_dir = 8
            start_rank = 1
            promotion_rank = 6
            enemy_pieces = board.black_pieces
        else:
            forward_dir = -8
            start_rank = 6
            promotion_rank = 1
            enemy_pieces = board.white_pieces
        
        # Process each pawn
        current_bb, square = pawn_bb.pop_ls1b()
        while square != -1:
            file, rank = square % 8, square // 8
            
            # Captures
            pawn_attacks = self.attack_tables.pawn_attacks[color][square]
            captures = pawn_attacks & enemy_pieces
            
            capture_bb, capture_square = captures.pop_ls1b()
            while capture_square != -1:
                captured_piece, _ = board.get_piece_at(capture_square)
                
                if rank == promotion_rank:
                    # Promotion captures
                    for promotion_piece in [Piece.QUEEN, Piece.ROOK, Piece.BISHOP, Piece.KNIGHT]:
                        moves.append(Move(
                            from_square=square,
                            to_square=capture_square,
                            piece=Piece.PAWN,
                            captured_piece=captured_piece,
                            promotion_piece=promotion_piece
                        ))
                else:
                    moves.append(Move(
                        from_square=square,
                        to_square=capture_square,
                        piece=Piece.PAWN,
                        captured_piece=captured_piece
                    ))
                
                capture_bb, capture_square = capture_bb.pop_ls1b()
            
            # En passant
            if board.en_passant_square is not None:
                en_passant_attacks = pawn_attacks & BitBoard(1 << board.en_passant_square)
                if en_passant_attacks:
                    moves.append(Move(
                        from_square=square,
                        to_square=board.en_passant_square,
                        piece=Piece.PAWN,
                        captured_piece=Piece.PAWN,
                        is_en_passant=True
                    ))
            
            if not only_captures:
                # Forward moves
                forward_square = square + forward_dir
                if 0 <= forward_square < 64 and not board.occupied.get_bit(forward_square):
                    if rank == promotion_rank:
                        # Promotions
                        for promotion_piece in [Piece.QUEEN, Piece.ROOK, Piece.BISHOP, Piece.KNIGHT]:
                            moves.append(Move(
                                from_square=square,
                                to_square=forward_square,
                                piece=Piece.PAWN,
                                promotion_piece=promotion_piece
                            ))
                    else:
                        moves.append(Move(
                            from_square=square,
                            to_square=forward_square,
                            piece=Piece.PAWN
                        ))
                        
                        # Double push from starting rank
                        if rank == start_rank:
                            double_forward = square + 2 * forward_dir
                            if 0 <= double_forward < 64 and not board.occupied.get_bit(double_forward):
                                moves.append(Move(
                                    from_square=square,
                                    to_square=double_forward,
                                    piece=Piece.PAWN
                                ))
            
            current_bb, square = current_bb.pop_ls1b()
        
        return moves
    
    def _generate_knight_moves(self, board: ChessBoard, color: Color, only_captures: bool) -> List[Move]:
        """Generate knight moves."""
        moves = []
        knight_bb = board.bitboards[(color, Piece.KNIGHT)]
        own_pieces = board.white_pieces if color == Color.WHITE else board.black_pieces
        enemy_pieces = board.black_pieces if color == Color.WHITE else board.white_pieces
        
        current_bb, square = knight_bb.pop_ls1b()
        while square != -1:
            attacks = self.attack_tables.knight_attacks[square]
            
            if only_captures:
                targets = attacks & enemy_pieces
            else:
                targets = attacks & ~own_pieces
            
            target_bb, target_square = targets.pop_ls1b()
            while target_square != -1:
                captured_piece, _ = board.get_piece_at(target_square)
                moves.append(Move(
                    from_square=square,
                    to_square=target_square,
                    piece=Piece.KNIGHT,
                    captured_piece=captured_piece
                ))
                target_bb, target_square = target_bb.pop_ls1b()
            
            current_bb, square = current_bb.pop_ls1b()
        
        return moves
    
    def _generate_sliding_moves(self, board: ChessBoard, color: Color, piece: Piece, 
                              only_captures: bool) -> List[Move]:
        """Generate moves for sliding pieces (bishop, rook, queen)."""
        moves = []
        piece_bb = board.bitboards[(color, piece)]
        own_pieces = board.white_pieces if color == Color.WHITE else board.black_pieces
        enemy_pieces = board.black_pieces if color == Color.WHITE else board.white_pieces
        
        current_bb, square = piece_bb.pop_ls1b()
        while square != -1:
            # Get attack pattern based on piece type
            if piece == Piece.BISHOP:
                attacks = self._get_bishop_attacks(square, board.occupied)
            elif piece == Piece.ROOK:
                attacks = self._get_rook_attacks(square, board.occupied)
            else:  # Queen
                attacks = (self._get_bishop_attacks(square, board.occupied) |
                          self._get_rook_attacks(square, board.occupied))
            
            if only_captures:
                targets = attacks & enemy_pieces
            else:
                targets = attacks & ~own_pieces
            
            target_bb, target_square = targets.pop_ls1b()
            while target_square != -1:
                captured_piece, _ = board.get_piece_at(target_square)
                moves.append(Move(
                    from_square=square,
                    to_square=target_square,
                    piece=piece,
                    captured_piece=captured_piece
                ))
                target_bb, target_square = target_bb.pop_ls1b()
            
            current_bb, square = current_bb.pop_ls1b()
        
        return moves
    
    def _generate_bishop_moves(self, board: ChessBoard, color: Color, only_captures: bool) -> List[Move]:
        """Generate bishop moves."""
        return self._generate_sliding_moves(board, color, Piece.BISHOP, only_captures)
    
    def _generate_rook_moves(self, board: ChessBoard, color: Color, only_captures: bool) -> List[Move]:
        """Generate rook moves."""
        return self._generate_sliding_moves(board, color, Piece.ROOK, only_captures)
    
    def _generate_queen_moves(self, board: ChessBoard, color: Color, only_captures: bool) -> List[Move]:
        """Generate queen moves."""
        return self._generate_sliding_moves(board, color, Piece.QUEEN, only_captures)
    
    def _generate_king_moves(self, board: ChessBoard, color: Color, only_captures: bool) -> List[Move]:
        """Generate king moves."""
        moves = []
        king_bb = board.bitboards[(color, Piece.KING)]
        own_pieces = board.white_pieces if color == Color.WHITE else board.black_pieces
        enemy_pieces = board.black_pieces if color == Color.WHITE else board.white_pieces
        
        current_bb, square = king_bb.pop_ls1b()
        while square != -1:
            attacks = self.attack_tables.king_attacks[square]
            
            if only_captures:
                targets = attacks & enemy_pieces
            else:
                targets = attacks & ~own_pieces
            
            target_bb, target_square = targets.pop_ls1b()
            while target_square != -1:
                captured_piece, _ = board.get_piece_at(target_square)
                moves.append(Move(
                    from_square=square,
                    to_square=target_square,
                    piece=Piece.KING,
                    captured_piece=captured_piece
                ))
                target_bb, target_square = target_bb.pop_ls1b()
            
            current_bb, square = current_bb.pop_ls1b()
        
        return moves
    
    def _generate_castling_moves(self, board: ChessBoard, color: Color) -> List[Move]:
        """Generate castling moves."""
        moves = []
        
        if color == Color.WHITE:
            king_square = Square.E1
            # Kingside castling
            if (board.castling_rights.white_kingside and
                not board.occupied.get_bit(Square.F1) and
                not board.occupied.get_bit(Square.G1) and
                not self._is_square_attacked(board, Square.E1, Color.BLACK) and
                not self._is_square_attacked(board, Square.F1, Color.BLACK) and
                not self._is_square_attacked(board, Square.G1, Color.BLACK)):
                moves.append(Move(
                    from_square=Square.E1,
                    to_square=Square.G1,
                    piece=Piece.KING,
                    is_castling=True
                ))
            
            # Queenside castling
            if (board.castling_rights.white_queenside and
                not board.occupied.get_bit(Square.B1) and
                not board.occupied.get_bit(Square.C1) and
                not board.occupied.get_bit(Square.D1) and
                not self._is_square_attacked(board, Square.E1, Color.BLACK) and
                not self._is_square_attacked(board, Square.D1, Color.BLACK) and
                not self._is_square_attacked(board, Square.C1, Color.BLACK)):
                moves.append(Move(
                    from_square=Square.E1,
                    to_square=Square.C1,
                    piece=Piece.KING,
                    is_castling=True
                ))
        else:
            # Black castling (similar logic)
            if (board.castling_rights.black_kingside and
                not board.occupied.get_bit(Square.F8) and
                not board.occupied.get_bit(Square.G8) and
                not self._is_square_attacked(board, Square.E8, Color.WHITE) and
                not self._is_square_attacked(board, Square.F8, Color.WHITE) and
                not self._is_square_attacked(board, Square.G8, Color.WHITE)):
                moves.append(Move(
                    from_square=Square.E8,
                    to_square=Square.G8,
                    piece=Piece.KING,
                    is_castling=True
                ))
            
            if (board.castling_rights.black_queenside and
                not board.occupied.get_bit(Square.B8) and
                not board.occupied.get_bit(Square.C8) and
                not board.occupied.get_bit(Square.D8) and
                not self._is_square_attacked(board, Square.E8, Color.WHITE) and
                not self._is_square_attacked(board, Square.D8, Color.WHITE) and
                not self._is_square_attacked(board, Square.C8, Color.WHITE)):
                moves.append(Move(
                    from_square=Square.E8,
                    to_square=Square.C8,
                    piece=Piece.KING,
                    is_castling=True
                ))
        
        return moves
    
    def _get_bishop_attacks(self, square: int, occupied: BitBoard) -> BitBoard:
        """Get bishop attack pattern for given square and occupancy."""
        attacks = BitBoard(0)
        file, rank = square % 8, square // 8
        
        # Diagonal directions
        directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        
        for df, dr in directions:
            f, r = file + df, rank + dr
            while 0 <= f < 8 and 0 <= r < 8:
                target_square = r * 8 + f
                attacks = attacks.set_bit(target_square)
                
                if occupied.get_bit(target_square):
                    break
                
                f += df
                r += dr
        
        return attacks
    
    def _get_rook_attacks(self, square: int, occupied: BitBoard) -> BitBoard:
        """Get rook attack pattern for given square and occupancy."""
        attacks = BitBoard(0)
        file, rank = square % 8, square // 8
        
        # Horizontal and vertical directions
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        
        for df, dr in directions:
            f, r = file + df, rank + dr
            while 0 <= f < 8 and 0 <= r < 8:
                target_square = r * 8 + f
                attacks = attacks.set_bit(target_square)
                
                if occupied.get_bit(target_square):
                    break
                
                f += df
                r += dr
        
        return attacks
    
    def _is_square_attacked(self, board: ChessBoard, square: int, by_color: Color) -> bool:
        """Check if a square is attacked by the given color."""
        # Check pawn attacks
        pawn_bb = board.bitboards[(by_color, Piece.PAWN)]
        enemy_color = Color.BLACK if by_color == Color.WHITE else Color.WHITE
        pawn_attacks = self.attack_tables.pawn_attacks[enemy_color][square]
        if pawn_attacks & pawn_bb:
            return True
        
        # Check knight attacks
        knight_bb = board.bitboards[(by_color, Piece.KNIGHT)]
        knight_attacks = self.attack_tables.knight_attacks[square]
        if knight_attacks & knight_bb:
            return True
        
        # Check bishop/queen diagonal attacks
        bishop_bb = board.bitboards[(by_color, Piece.BISHOP)]
        queen_bb = board.bitboards[(by_color, Piece.QUEEN)]
        diagonal_attacks = self._get_bishop_attacks(square, board.occupied)
        if diagonal_attacks & (bishop_bb | queen_bb):
            return True
        
        # Check rook/queen horizontal/vertical attacks
        rook_bb = board.bitboards[(by_color, Piece.ROOK)]
        rank_file_attacks = self._get_rook_attacks(square, board.occupied)
        if rank_file_attacks & (rook_bb | queen_bb):
            return True
        
        # Check king attacks
        king_bb = board.bitboards[(by_color, Piece.KING)]
        king_attacks = self.attack_tables.king_attacks[square]
        if king_attacks & king_bb:
            return True
        
        return False
    
    def _is_legal_move(self, board: ChessBoard, move: Move) -> bool:
        """Check if a move is legal (doesn't leave own king in check)."""
        # Make the move on a copy of the board
        board_copy = board.copy()
        board_copy.make_move(move)
        
        # Find the king of the player who just moved
        original_color = Color.BLACK if board_copy.to_move == Color.WHITE else Color.WHITE
        king_bb = board_copy.bitboards[(original_color, Piece.KING)]
        _, king_square = king_bb.pop_ls1b()
        
        # Check if king is in check
        return not self._is_square_attacked(board_copy, king_square, board_copy.to_move)
    
    def is_in_check(self, board: ChessBoard, color: Color) -> bool:
        """Check if the given color's king is in check."""
        king_bb = board.bitboards[(color, Piece.KING)]
        _, king_square = king_bb.pop_ls1b()
        
        if king_square == -1:
            return False  # No king found
        
        enemy_color = Color.BLACK if color == Color.WHITE else Color.WHITE
        return self._is_square_attacked(board, king_square, enemy_color)
    
    def is_checkmate(self, board: ChessBoard) -> bool:
        """Check if current position is checkmate."""
        if not self.is_in_check(board, board.to_move):
            return False
        
        legal_moves = self.generate_moves(board)
        return len(legal_moves) == 0
    
    def is_stalemate(self, board: ChessBoard) -> bool:
        """Check if current position is stalemate."""
        if self.is_in_check(board, board.to_move):
            return False
        
        legal_moves = self.generate_moves(board)
        return len(legal_moves) == 0
    
    def _generate_pawn_moves_fast(self, board: ChessBoard, color: Color, only_captures: bool, moves: List[Move]) -> None:
        """Optimized pawn move generation that appends directly to moves list."""
        pawn_bb = board.bitboards[(color, Piece.PAWN)]
        if not pawn_bb.value:
            return
            
        if color == Color.WHITE:
            forward_dir = 8
            start_rank = 1
            promotion_rank = 6
            enemy_pieces = board.black_pieces
        else:
            forward_dir = -8
            start_rank = 6
            promotion_rank = 1
            enemy_pieces = board.white_pieces
        
        # Process each pawn - optimized loop
        squares = []
        temp_bb = pawn_bb
        while temp_bb.value:
            temp_bb, square = temp_bb.pop_ls1b()
            squares.append(square)
        
        for square in squares:
            rank = square // 8
            
            # Captures - pre-computed attack table lookup
            pawn_attacks = self.attack_tables.pawn_attacks[color][square]
            captures = pawn_attacks & enemy_pieces
            
            # Process captures
            if captures.value:
                capture_squares = []
                temp_bb = captures
                while temp_bb.value:
                    temp_bb, capture_square = temp_bb.pop_ls1b()
                    capture_squares.append(capture_square)
                
                for capture_square in capture_squares:
                    captured_piece, _ = board.get_piece_at(capture_square)
                    
                    if rank == promotion_rank:
                        # Promotion captures
                        for promotion_piece in [Piece.QUEEN, Piece.ROOK, Piece.BISHOP, Piece.KNIGHT]:
                            moves.append(Move(
                                from_square=square,
                                to_square=capture_square,
                                piece=Piece.PAWN,
                                captured_piece=captured_piece,
                                promotion_piece=promotion_piece
                            ))
                    else:
                        moves.append(Move(
                            from_square=square,
                            to_square=capture_square,
                            piece=Piece.PAWN,
                            captured_piece=captured_piece
                        ))
            
            # En passant
            if board.en_passant_square is not None:
                en_passant_attacks = pawn_attacks & BitBoard(1 << board.en_passant_square)
                if en_passant_attacks.value:
                    moves.append(Move(
                        from_square=square,
                        to_square=board.en_passant_square,
                        piece=Piece.PAWN,
                        captured_piece=Piece.PAWN,
                        is_en_passant=True
                    ))
            
            if not only_captures:
                # Forward moves
                forward_square = square + forward_dir
                if 0 <= forward_square < 64 and not board.occupied.get_bit(forward_square):
                    if rank == promotion_rank:
                        # Promotions
                        for promotion_piece in [Piece.QUEEN, Piece.ROOK, Piece.BISHOP, Piece.KNIGHT]:
                            moves.append(Move(
                                from_square=square,
                                to_square=forward_square,
                                piece=Piece.PAWN,
                                promotion_piece=promotion_piece
                            ))
                    else:
                        moves.append(Move(
                            from_square=square,
                            to_square=forward_square,
                            piece=Piece.PAWN
                        ))
                          # Double pawn push
                        if rank == start_rank:
                            double_square = square + 2 * forward_dir
                            if 0 <= double_square < 64 and not board.occupied.get_bit(double_square):
                                moves.append(Move(
                                    from_square=square,
                                    to_square=double_square,
                                    piece=Piece.PAWN
                                ))

    def _generate_knight_moves_fast(self, board: ChessBoard, color: Color, only_captures: bool, moves: List[Move]) -> None:
        """Optimized knight move generation."""
        knight_bb = board.bitboards[(color, Piece.KNIGHT)]
        if not knight_bb.value:
            return
            
        own_pieces = board.white_pieces if color == Color.WHITE else board.black_pieces
        enemy_pieces = board.black_pieces if color == Color.WHITE else board.white_pieces
        
        squares = []
        temp_bb = knight_bb
        while temp_bb.value:
            temp_bb, square = temp_bb.pop_ls1b()
            squares.append(square)
        
        for square in squares:
            attacks = self.attack_tables.knight_attacks[square]
            
            if only_captures:
                targets = attacks & enemy_pieces
            else:
                targets = attacks & ~own_pieces
            
            if targets.value:
                target_squares = []
                temp_bb = targets
                while temp_bb.value:
                    temp_bb, target_square = temp_bb.pop_ls1b()
                    target_squares.append(target_square)
                
                for target_square in target_squares:
                    captured_piece, _ = board.get_piece_at(target_square)
                    moves.append(Move(
                        from_square=square,
                        to_square=target_square,
                        piece=Piece.KNIGHT,
                        captured_piece=captured_piece
                    ))

    def _generate_sliding_moves_fast(self, board: ChessBoard, color: Color, only_captures: bool, moves: List[Move]) -> None:
        """Optimized sliding piece move generation (bishop, rook, queen)."""
        own_pieces = board.white_pieces if color == Color.WHITE else board.black_pieces
        enemy_pieces = board.black_pieces if color == Color.WHITE else board.white_pieces
        
        # Bishops
        bishop_bb = board.bitboards[(color, Piece.BISHOP)]
        self._generate_piece_moves_fast(board, bishop_bb, Piece.BISHOP, own_pieces, enemy_pieces, only_captures, moves, True)
        
        # Rooks
        rook_bb = board.bitboards[(color, Piece.ROOK)]
        self._generate_piece_moves_fast(board, rook_bb, Piece.ROOK, own_pieces, enemy_pieces, only_captures, moves, False)
        
        # Queens (both diagonal and orthogonal)
        queen_bb = board.bitboards[(color, Piece.QUEEN)]
        self._generate_piece_moves_fast(board, queen_bb, Piece.QUEEN, own_pieces, enemy_pieces, only_captures, moves, True)
        self._generate_piece_moves_fast(board, queen_bb, Piece.QUEEN, own_pieces, enemy_pieces, only_captures, moves, False)

    def _generate_piece_moves_fast(self, board: ChessBoard, piece_bb: BitBoard, piece_type: Piece, 
                                  own_pieces: BitBoard, enemy_pieces: BitBoard, only_captures: bool, 
                                  moves: List[Move], is_diagonal: bool) -> None:
        """Generate moves for a sliding piece type."""
        if not piece_bb.value:
            return
            
        squares = []
        temp_bb = piece_bb
        while temp_bb.value:
            temp_bb, square = temp_bb.pop_ls1b()
            squares.append(square)
        
        for square in squares:
            if is_diagonal:
                attacks = self._get_bishop_attacks(square, board.occupied)
            else:
                attacks = self._get_rook_attacks(square, board.occupied)
            
            if only_captures:
                targets = attacks & enemy_pieces
            else:
                targets = attacks & ~own_pieces
            
            if targets.value:
                target_squares = []
                temp_bb = targets
                while temp_bb.value:
                    temp_bb, target_square = temp_bb.pop_ls1b()
                    target_squares.append(target_square)
                
                for target_square in target_squares:
                    captured_piece, _ = board.get_piece_at(target_square)
                    moves.append(Move(
                        from_square=square,
                        to_square=target_square,
                        piece=piece_type,
                        captured_piece=captured_piece
                    ))

    def _generate_king_moves_fast(self, board: ChessBoard, color: Color, only_captures: bool, moves: List[Move]) -> None:
        """Optimized king move generation."""
        king_bb = board.bitboards[(color, Piece.KING)]
        if not king_bb.value:
            return
            
        own_pieces = board.white_pieces if color == Color.WHITE else board.black_pieces
        enemy_pieces = board.black_pieces if color == Color.WHITE else board.white_pieces
        
        _, square = king_bb.pop_ls1b()
        if square != -1:
            attacks = self.attack_tables.king_attacks[square]
            
            if only_captures:
                targets = attacks & enemy_pieces
            else:
                targets = attacks & ~own_pieces
            
            if targets.value:
                target_squares = []
                temp_bb = targets
                while temp_bb.value:
                    temp_bb, target_square = temp_bb.pop_ls1b()
                    target_squares.append(target_square)
                
                for target_square in target_squares:
                    captured_piece, _ = board.get_piece_at(target_square)
                    moves.append(Move(
                        from_square=square,
                        to_square=target_square,
                        piece=Piece.KING,
                        captured_piece=captured_piece
                    ))

    def _generate_castling_moves_fast(self, board: ChessBoard, color: Color, moves: List[Move]) -> None:
        """Optimized castling move generation."""
        if color == Color.WHITE:
            king_square = Square.E1
            # Kingside castling
            if (board.castling_rights.white_kingside and
                not board.occupied.get_bit(Square.F1) and
                not board.occupied.get_bit(Square.G1) and
                not self._is_square_attacked(board, Square.E1, Color.BLACK) and
                not self._is_square_attacked(board, Square.F1, Color.BLACK) and
                not self._is_square_attacked(board, Square.G1, Color.BLACK)):
                moves.append(Move(
                    from_square=Square.E1,
                    to_square=Square.G1,
                    piece=Piece.KING,
                    is_castling=True
                ))
            
            # Queenside castling
            if (board.castling_rights.white_queenside and
                not board.occupied.get_bit(Square.D1) and
                not board.occupied.get_bit(Square.C1) and
                not board.occupied.get_bit(Square.B1) and
                not self._is_square_attacked(board, Square.E1, Color.BLACK) and
                not self._is_square_attacked(board, Square.D1, Color.BLACK) and
                not self._is_square_attacked(board, Square.C1, Color.BLACK)):
                moves.append(Move(
                    from_square=Square.E1,
                    to_square=Square.C1,
                    piece=Piece.KING,
                    is_castling=True
                ))
        else:
            # Black castling
            if (board.castling_rights.black_kingside and
                not board.occupied.get_bit(Square.F8) and
                not board.occupied.get_bit(Square.G8) and
                not self._is_square_attacked(board, Square.E8, Color.WHITE) and
                not self._is_square_attacked(board, Square.F8, Color.WHITE) and
                not self._is_square_attacked(board, Square.G8, Color.WHITE)):
                moves.append(Move(
                    from_square=Square.E8,
                    to_square=Square.G8,
                    piece=Piece.KING,
                    is_castling=True
                ))
            
            if (board.castling_rights.black_queenside and
                not board.occupied.get_bit(Square.D8) and
                not board.occupied.get_bit(Square.C8) and
                not board.occupied.get_bit(Square.B8) and
                not self._is_square_attacked(board, Square.E8, Color.WHITE) and
                not self._is_square_attacked(board, Square.D8, Color.WHITE) and
                not self._is_square_attacked(board, Square.C8, Color.WHITE)):
                moves.append(Move(
                    from_square=Square.E8,
                    to_square=Square.C8,                    piece=Piece.KING,
                    is_castling=True
                ))
    
    def _is_legal_move_fast(self, board: ChessBoard, move: Move, king_square: int) -> bool:
        """Optimized legal move check."""
        # For king moves, we need to check if the destination square is attacked
        if move.piece == Piece.KING:
            enemy_color = Color.BLACK if board.to_move == Color.WHITE else Color.WHITE
            return not self._is_square_attacked(board, move.to_square, enemy_color)
        
        # For other pieces, check if king would be in check after the move
        # This is still expensive but necessary - we could optimize further with pin detection
        board_copy = board.copy()
        board_copy.make_move(move)
        
        original_color = Color.BLACK if board_copy.to_move == Color.WHITE else Color.WHITE
        king_bb = board_copy.bitboards[(original_color, Piece.KING)]
        _, new_king_square = king_bb.pop_ls1b()
        
        return not self._is_square_attacked(board_copy, new_king_square, board_copy.to_move)
    
    def _get_pinned_pieces(self, board: ChessBoard, king_square: int, opponent_color: Color) -> set:
        """Get set of pieces that are pinned to the king."""
        pinned = set()
        
        # Get all enemy sliding pieces (bishop, rook, queen)
        enemy_bishops = board.bitboards[(opponent_color, Piece.BISHOP)]
        enemy_rooks = board.bitboards[(opponent_color, Piece.ROOK)]
        enemy_queens = board.bitboards[(opponent_color, Piece.QUEEN)]
        
        # Check for pins along diagonals (bishop and queen)
        diagonals = enemy_bishops | enemy_queens
        while diagonals.value:
            diagonals, piece_square = diagonals.pop_ls1b()
            pin_square = self._check_pin_ray(board, king_square, piece_square, True)
            if pin_square != -1:
                pinned.add(pin_square)
        
        # Check for pins along ranks/files (rook and queen)  
        orthogonals = enemy_rooks | enemy_queens
        while orthogonals.value:
            orthogonals, piece_square = orthogonals.pop_ls1b()
            pin_square = self._check_pin_ray(board, king_square, piece_square, False)
            if pin_square != -1:
                pinned.add(pin_square)
        
        return pinned
    
    def _check_pin_ray(self, board: ChessBoard, king_square: int, attacker_square: int, is_diagonal: bool) -> int:
        """Check if there's a pin along the ray from attacker to king."""
        king_file, king_rank = king_square % 8, king_square // 8
        att_file, att_rank = attacker_square % 8, attacker_square // 8
        
        # Determine direction
        if is_diagonal:
            if abs(king_file - att_file) != abs(king_rank - att_rank):
                return -1  # Not on diagonal
            df = 1 if king_file > att_file else -1
            dr = 1 if king_rank > att_rank else -1
        else:
            if king_file != att_file and king_rank != att_rank:
                return -1  # Not on rank or file
            df = 0 if king_file == att_file else (1 if king_file > att_file else -1)
            dr = 0 if king_rank == att_rank else (1 if king_rank > att_rank else -1)
        
        # Walk from attacker towards king, looking for exactly one piece
        f, r = att_file + df, att_rank + dr
        pieces_found = 0
        pinned_square = -1
        
        while f != king_file or r != king_rank:
            square = r * 8 + f
            if board.occupied.get_bit(square):
                pieces_found += 1
                if pieces_found == 1:
                    pinned_square = square
                elif pieces_found > 1:
                    return -1  # No pin if more than one piece
            f += df
            r += dr
        
        return pinned_square if pieces_found == 1 else -1
    
    def _is_legal_move_optimized(self, board: ChessBoard, move: Move, king_square: int, pinned_pieces: set) -> bool:
        """Optimized legal move check using pin detection."""
        # King moves always need full check
        if move.piece == Piece.KING:
            enemy_color = Color.BLACK if board.to_move == Color.WHITE else Color.WHITE
            return not self._is_square_attacked(board, move.to_square, enemy_color)
        
        # If piece is not pinned, move is legal
        if move.from_square not in pinned_pieces:
            return True
        
        # If piece is pinned, it can only move along the pin ray
        return self._move_along_pin_ray(king_square, move.from_square, move.to_square)
    
    def _move_along_pin_ray(self, king_square: int, from_square: int, to_square: int) -> bool:
        """Check if move is along the pin ray (king-from_square line)."""
        king_file, king_rank = king_square % 8, king_square // 8
        from_file, from_rank = from_square % 8, from_square // 8
        to_file, to_rank = to_square % 8, to_square // 8
        
        # Check if all three squares are collinear
        # Vector from king to from_square
        df1, dr1 = from_file - king_file, from_rank - king_rank
        # Vector from king to to_square  
        df2, dr2 = to_file - king_file, to_rank - king_rank
        
        # Check collinearity: cross product should be zero
        return df1 * dr2 == df2 * dr1
    
    def _is_legal_move_when_in_check(self, board: ChessBoard, move: Move, king_square: int, opponent_color: Color) -> bool:
        """Legal move check when king is in check - must block, capture, or move king."""
        # King moves need to be to safe squares
        if move.piece == Piece.KING:
            return not self._is_square_attacked(board, move.to_square, opponent_color)
        
        # Other moves must block the check or capture the checking piece
        # For simplicity, fall back to full check for now
        # TODO: Implement proper check evasion logic
        board_copy = board.copy()
        board_copy.make_move(move)
        
        original_color = Color.BLACK if board_copy.to_move == Color.WHITE else Color.WHITE
        king_bb = board_copy.bitboards[(original_color, Piece.KING)]
        _, new_king_square = king_bb.pop_ls1b()
        
        return not self._is_square_attacked(board_copy, new_king_square, board_copy.to_move)
