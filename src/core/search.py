"""
Chess Search Algorithms
======================

Implementation of Alpha-Beta search with advanced pruning techniques.
This forms the core of the chess engine's tactical strength.
"""

from typing import Optional, List, Tuple, Dict, Any
import time
import math
from dataclasses import dataclass
from .board import ChessBoard, Move, Color, Piece
from .moves import MoveGenerator
from .evaluation import Evaluator


@dataclass
class SearchInfo:
    """Information about the current search."""
    nodes: int = 0
    depth: int = 0
    time_start: float = 0.0
    time_limit: float = 0.0
    stopped: bool = False
    pv: List[Move] = None
    
    def __post_init__(self):
        if self.pv is None:
            self.pv = []


@dataclass
class TranspositionEntry:
    """Entry in the transposition table."""
    zobrist_key: int
    depth: int
    score: int
    flag: int  # 0=exact, 1=lower_bound, 2=upper_bound
    best_move: Optional[Move] = None
    age: int = 0


class TranspositionTable:
    """Hash table for storing previously searched positions."""
    
    EXACT = 0
    LOWER_BOUND = 1
    UPPER_BOUND = 2
    
    def __init__(self, size_mb: int = 64):
        """Initialize transposition table with given size in MB."""
        self.size = (size_mb * 1024 * 1024) // 32  # Rough estimate of entry size
        self.table: Dict[int, TranspositionEntry] = {}
        self.age = 0
    
    def store(self, zobrist_key: int, depth: int, score: int, flag: int, 
              best_move: Optional[Move] = None):
        """Store position in transposition table."""
        if len(self.table) >= self.size:
            # Simple replacement: remove oldest entries
            oldest_keys = [k for k, v in self.table.items() 
                          if v.age < self.age - 2]
            for key in oldest_keys[:len(oldest_keys)//2]:
                del self.table[key]
        
        self.table[zobrist_key] = TranspositionEntry(
            zobrist_key=zobrist_key,
            depth=depth,
            score=score,
            flag=flag,
            best_move=best_move,
            age=self.age
        )
    
    def probe(self, zobrist_key: int) -> Optional[TranspositionEntry]:
        """Probe transposition table for position."""
        return self.table.get(zobrist_key)
    
    def clear(self):
        """Clear transposition table."""
        self.table.clear()
        self.age += 1


class SearchEngine:
    """Main search engine implementing Alpha-Beta with enhancements."""
    
    def __init__(self, evaluator: Evaluator, move_generator: MoveGenerator):
        """Initialize search engine."""
        self.evaluator = evaluator
        self.move_generator = move_generator
        self.transposition_table = TranspositionTable()
        
        # Search parameters
        self.max_depth = 64
        self.max_quiesce_depth = 16
        
        # Killer moves (heuristic for move ordering)
        self.killer_moves: List[List[Optional[Move]]] = [
            [None, None] for _ in range(self.max_depth)
        ]
        
        # History heuristic
        self.history_scores: Dict[Tuple[int, int], int] = {}
    
    def search(self, board: ChessBoard, depth: int, time_limit: float = None) -> Tuple[Move, int, SearchInfo]:
        """
        Main search function using iterative deepening.
        
        Args:
            board: Current board position
            depth: Maximum search depth
            time_limit: Time limit in seconds
            
        Returns:
            Tuple of (best_move, score, search_info)
        """
        info = SearchInfo()
        info.time_start = time.time()
        info.time_limit = time_limit or float('inf')
        
        best_move = None
        best_score = float('-inf')
        
        # Iterative deepening
        for current_depth in range(1, depth + 1):
            if self._should_stop(info):
                break
                
            info.depth = current_depth
            score, move, pv = self._alpha_beta_root(board, current_depth, info)
            
            if not self._should_stop(info):
                best_move = move
                best_score = score
                info.pv = pv
                
                # Print info (for UCI)
                elapsed = time.time() - info.time_start
                nps = info.nodes / max(elapsed, 0.001)
                print(f"info depth {current_depth} score cp {score} "
                      f"nodes {info.nodes} nps {int(nps)} time {int(elapsed*1000)} "
                      f"pv {' '.join(str(m) for m in pv[:5])}")
        
        return best_move, best_score, info
    
    def _alpha_beta_root(self, board: ChessBoard, depth: int, info: SearchInfo) -> Tuple[int, Move, List[Move]]:
        """Root alpha-beta search."""
        alpha = float('-inf')
        beta = float('inf')
        best_move = None
        best_pv = []
        
        moves = list(self.move_generator.generate_legal_moves(board))
        moves = self._order_moves(board, moves, depth)
        
        for move in moves:
            if self._should_stop(info):
                break
            board_copy = board.copy()
            board_copy.make_move(move)
            score, pv = self._alpha_beta(board_copy, depth - 1, -beta, -alpha, info, False)
            score = -score
            
            if score > alpha:
                alpha = score
                best_move = move
                best_pv = [move] + pv
        
        return alpha, best_move, best_pv
    
    def _alpha_beta(self, board: ChessBoard, depth: int, alpha: float, beta: float, 
                   info: SearchInfo, is_pv: bool) -> Tuple[int, List[Move]]:
        """
        Alpha-Beta search with pruning.
        
        Args:
            board: Current position
            depth: Remaining search depth
            alpha: Alpha value
            beta: Beta value
            info: Search information
            is_pv: Whether this is a PV node
            
        Returns:
            Tuple of (score, principal_variation)
        """
        info.nodes += 1
        
        if self._should_stop(info):
            return 0, []
        
        # Check for draw conditions
        if board.is_repetition() or board.halfmove_clock >= 100:
            return 0, []
        
        # Probe transposition table
        tt_entry = self.transposition_table.probe(board.zobrist_hash)
        tt_move = None
        
        if tt_entry and tt_entry.depth >= depth:
            if tt_entry.flag == TranspositionTable.EXACT:
                return tt_entry.score, []
            elif tt_entry.flag == TranspositionTable.LOWER_BOUND:
                alpha = max(alpha, tt_entry.score)
            elif tt_entry.flag == TranspositionTable.UPPER_BOUND:
                beta = min(beta, tt_entry.score)
            
            if alpha >= beta:
                return tt_entry.score, []
            
            tt_move = tt_entry.best_move
        
        # Terminal node evaluation
        if depth <= 0:
            return self._quiescence_search(board, alpha, beta, info, 0), []
        
        # Check for checkmate/stalemate
        moves = list(self.move_generator.generate_legal_moves(board))
        if not moves:
            if board.is_in_check():
                return -20000 + (self.max_depth - depth), []  # Checkmate
            else:
                return 0, []  # Stalemate
        
        # Move ordering
        moves = self._order_moves(board, moves, depth, tt_move)
        
        best_score = float('-inf')
        best_pv = []
        best_move = None
        flag = TranspositionTable.UPPER_BOUND
        
        for i, move in enumerate(moves):
            if self._should_stop(info):
                break
            
            board_copy = board.copy()
            board_copy.make_move(move)
            
            # Late Move Reductions (LMR)
            reduction = 0
            if (depth >= 3 and i >= 4 and not is_pv and 
                not board_copy.is_in_check() and not move.captured_piece and 
                not move.promotion_piece):
                reduction = 1 + (i > 8)
            
            # Search with reduction
            if reduction > 0:
                score, pv = self._alpha_beta(board_copy, depth - 1 - reduction, 
                                           -(alpha + 1), -alpha, info, False)
                score = -score
                
                # Re-search if reduction failed
                if score > alpha:
                    score, pv = self._alpha_beta(board_copy, depth - 1, -beta, -alpha, info, False)
                    score = -score
            else:
                # Principal Variation Search (PVS)
                if i == 0:
                    score, pv = self._alpha_beta(board_copy, depth - 1, -beta, -alpha, info, is_pv)
                    score = -score
                else:
                    # Null window search
                    score, _ = self._alpha_beta(board_copy, depth - 1, -(alpha + 1), -alpha, info, False)
                    score = -score
                    
                    # Re-search if null window failed
                    if alpha < score < beta and is_pv:
                        score, pv = self._alpha_beta(board_copy, depth - 1, -beta, -alpha, info, True)
                        score = -score
            
            if score > best_score:
                best_score = score
                best_pv = [move] + pv
                best_move = move
                
                if score > alpha:
                    alpha = score
                    flag = TranspositionTable.EXACT
                    
                    # Update killer moves
                    if not move.captured_piece and depth < len(self.killer_moves):
                        killers = self.killer_moves[depth]
                        if killers[0] != move:
                            killers[1] = killers[0]
                            killers[0] = move
                
                if alpha >= beta:
                    flag = TranspositionTable.LOWER_BOUND
                    
                    # Update history heuristic
                    if not move.captured_piece:
                        key = (move.from_square, move.to_square)
                        self.history_scores[key] = self.history_scores.get(key, 0) + depth * depth
                    
                    break
        
        # Store in transposition table
        self.transposition_table.store(board.zobrist_hash, depth, best_score, flag, best_move)
        
        return best_score, best_pv
    
    def _quiescence_search(self, board: ChessBoard, alpha: float, beta: float, 
                          info: SearchInfo, ply: int) -> int:
        """
        Quiescence search to avoid horizon effect.
        Only searches captures, checks, and promotions.
        """
        info.nodes += 1
        
        if self._should_stop(info) or ply >= self.max_quiesce_depth:
            return self.evaluator.evaluate(board)
        
        # Stand pat evaluation
        stand_pat = self.evaluator.evaluate(board)
        
        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat
          # Generate and search tactical moves
        moves = list(self.move_generator.generate_captures(board))
        moves = self._order_captures(board, moves)
        
        for move in moves:
            # Delta pruning - skip hopeless captures
            if (stand_pat + self.evaluator.piece_values.get(move.captured_piece, 0) + 200 < alpha):
                continue
            
            board_copy = board.copy()
            board_copy.make_move(move)
            score = -self._quiescence_search(board_copy, -beta, -alpha, info, ply + 1)
            
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        
        return alpha
    
    def _order_moves(self, board: ChessBoard, moves: List[Move], depth: int, 
                    tt_move: Optional[Move] = None) -> List[Move]:
        """Order moves for better alpha-beta pruning."""
        def move_score(move: Move) -> int:
            score = 0
            
            # Transposition table move first
            if tt_move and move == tt_move:
                return 10000
            
            # Captures (MVV-LVA: Most Valuable Victim - Least Valuable Attacker)
            if move.captured_piece:
                score += 1000 + self.evaluator.piece_values[move.captured_piece] - move.piece.value
            
            # Promotions
            if move.promotion_piece:
                score += 900 + self.evaluator.piece_values[move.promotion_piece]
            
            # Killer moves
            if depth < len(self.killer_moves):
                killers = self.killer_moves[depth]
                if move == killers[0]:
                    score += 800
                elif move == killers[1]:
                    score += 700
            
            # History heuristic
            key = (move.from_square, move.to_square)
            score += self.history_scores.get(key, 0) // 100
            
            return score
        
        return sorted(moves, key=move_score, reverse=True)
    
    def _order_captures(self, board: ChessBoard, moves: List[Move]) -> List[Move]:
        """Order captures by MVV-LVA."""
        def capture_score(move: Move) -> int:
            if not move.captured_piece:
                return 0
            return (self.evaluator.piece_values[move.captured_piece] * 100 - 
                   self.evaluator.piece_values[move.piece])
        
        return sorted(moves, key=capture_score, reverse=True)
    
    def _should_stop(self, info: SearchInfo) -> bool:
        """Check if search should be stopped."""
        if info.stopped:
            return True
        
        if info.time_limit and time.time() - info.time_start > info.time_limit:
            info.stopped = True
            return True
        
        return False
    
    def stop(self):
        """Stop the current search."""
        # This would be called from another thread in a real implementation
        pass
