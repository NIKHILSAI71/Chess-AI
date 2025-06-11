#!/usr/bin/env python3
"""
Advanced Alpha-Beta Search Engine
===============================
State-of-the-art search implementation featuring:
- Principal Variation Search (PVS)
- Advanced pruning techniques (NMP, LMR, Futility)
- Sophisticated move ordering
- Transposition tables with Zobrist hashing
- Quiescence search
- Iterative deepening
- Parallel search capabilities
"""

import time
import math
import threading
from typing import Dict, List, Optional, Tuple, Any, NamedTuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from collections import defaultdict

# Search result structure
class SearchResult(NamedTuple):
    best_move: Any
    evaluation: float
    depth: int
    nodes_searched: int
    time_elapsed: float
    principal_variation: List[Any]
    search_info: Dict[str, Any]

# Transposition table entry types
class TTEntryType(Enum):
    EXACT = 0      # Exact evaluation
    LOWER_BOUND = 1 # Alpha cutoff (fail-high)
    UPPER_BOUND = 2 # Beta cutoff (fail-low)

@dataclass
class TranspositionEntry:
    """Transposition table entry"""
    zobrist_hash: int
    depth: int
    evaluation: float
    entry_type: TTEntryType
    best_move: Any
    age: int

class AdvancedSearchEngine:
    """
    Advanced alpha-beta search engine with cutting-edge optimizations
    """
    
    def __init__(self, 
                 evaluator=None,
                 move_generator=None,
                 zobrist_hasher=None,
                 max_depth: int = 10,
                 max_time: float = 5.0,
                 tt_size_mb: int = 128,
                 aspiration_window: int = 50,
                 use_parallel: bool = False,
                 num_threads: int = 4):
        
        self.evaluator = evaluator
        self.move_generator = move_generator
        self.zobrist_hasher = zobrist_hasher
        self.max_depth = max_depth
        self.max_time = max_time
        self.aspiration_window = aspiration_window
        self.use_parallel = use_parallel
        self.num_threads = num_threads
        
        # Transposition table
        tt_entries = (tt_size_mb * 1024 * 1024) // 64  # Estimate 64 bytes per entry
        self.transposition_table: Dict[int, TranspositionEntry] = {}
        self.tt_max_size = tt_entries
        self.tt_age = 0
        
        # Search statistics
        self.nodes_searched = 0
        self.tt_hits = 0
        self.tt_cutoffs = 0
        self.beta_cutoffs = 0
        self.start_time = 0.0
        
        # Move ordering tables
        self.killer_moves: Dict[int, List[Any]] = defaultdict(lambda: [None, None])
        self.history_table: Dict[Tuple[Any, Any], int] = defaultdict(int)
        self.counter_moves: Dict[Any, Any] = {}
        
        # Principal variation
        self.pv_table: Dict[int, List[Any]] = defaultdict(list)
        
        # Search control
        self.stop_search = False
        self.search_lock = threading.Lock()
        
        # Late move reduction parameters
        self.lmr_depth_threshold = 3
        self.lmr_move_threshold = 4
        self.lmr_reduction_base = 0.75
        
        # Null move parameters
        self.null_move_depth_threshold = 3
        self.null_move_reduction = 2
        
        # Futility pruning margins
        self.futility_margins = [0, 200, 300, 500, 800, 1200, 1800]
    
    def search(self, board, depth: Optional[int] = None, time_limit: Optional[float] = None) -> SearchResult:
        """
        Main search function with iterative deepening
        
        Args:
            board: Chess board position
            depth: Maximum search depth
            time_limit: Time limit in seconds
            
        Returns:
            SearchResult with best move and analysis
        """
        self.start_time = time.time()
        self.stop_search = False
        self.nodes_searched = 0
        self.tt_hits = 0
        self.tt_cutoffs = 0
        self.beta_cutoffs = 0
        
        depth = depth or self.max_depth
        time_limit = time_limit or self.max_time
        
        best_move = None
        best_evaluation = -float('inf')
        principal_variation = []
        
        # Clear PV table
        self.pv_table.clear()
        
        # Iterative deepening
        for current_depth in range(1, depth + 1):
            if self.should_stop_search(time_limit):
                break
            
            try:
                # Aspiration window search
                if current_depth >= 5 and best_evaluation != -float('inf'):
                    alpha = best_evaluation - self.aspiration_window
                    beta = best_evaluation + self.aspiration_window
                    
                    evaluation = self.principal_variation_search(
                        board, current_depth, alpha, beta, True, 0
                    )
                    
                    # Re-search if outside window
                    if evaluation <= alpha or evaluation >= beta:
                        evaluation = self.principal_variation_search(
                            board, current_depth, -float('inf'), float('inf'), True, 0
                        )
                else:
                    # Full window search
                    evaluation = self.principal_variation_search(
                        board, current_depth, -float('inf'), float('inf'), True, 0
                    )
                
                # Update best move and evaluation
                if len(self.pv_table[0]) > 0:
                    best_move = self.pv_table[0][0]
                    best_evaluation = evaluation
                    principal_variation = self.pv_table[0].copy()
                
                # Check time after each depth
                if self.should_stop_search(time_limit):
                    break
                    
            except Exception as e:
                print(f"Search error at depth {current_depth}: {e}")
                break
        
        elapsed_time = time.time() - self.start_time
        
        return SearchResult(
            best_move=best_move,
            evaluation=best_evaluation,
            depth=current_depth - 1,
            nodes_searched=self.nodes_searched,
            time_elapsed=elapsed_time,
            principal_variation=principal_variation,
            search_info={
                'tt_hits': self.tt_hits,
                'tt_cutoffs': self.tt_cutoffs,
                'beta_cutoffs': self.beta_cutoffs,
                'nps': self.nodes_searched / max(elapsed_time, 0.001)
            }
        )
    
    def principal_variation_search(self, 
                                 board, 
                                 depth: int, 
                                 alpha: float, 
                                 beta: float, 
                                 is_pv_node: bool,
                                 ply: int) -> float:
        """
        Principal Variation Search (PVS) implementation
        
        Args:
            board: Current board position
            depth: Remaining search depth
            alpha: Alpha bound
            beta: Beta bound
            is_pv_node: Whether this is a PV node
            ply: Current ply from root
            
        Returns:
            Position evaluation
        """
        # Clear PV at this ply
        self.pv_table[ply] = []
        
        # Check for search termination
        if self.should_stop_search():
            return 0.0
        
        self.nodes_searched += 1
        
        # Check transposition table
        tt_entry = self.probe_transposition_table(board)
        if tt_entry and tt_entry.depth >= depth and not is_pv_node:
            self.tt_hits += 1
            
            if tt_entry.entry_type == TTEntryType.EXACT:
                return tt_entry.evaluation
            elif tt_entry.entry_type == TTEntryType.LOWER_BOUND and tt_entry.evaluation >= beta:
                self.tt_cutoffs += 1
                return tt_entry.evaluation
            elif tt_entry.entry_type == TTEntryType.UPPER_BOUND and tt_entry.evaluation <= alpha:
                self.tt_cutoffs += 1
                return tt_entry.evaluation
        
        # Terminal node check
        if depth <= 0:
            return self.quiescence_search(board, alpha, beta, ply)
        
        # Null move pruning
        if (not is_pv_node and 
            depth >= self.null_move_depth_threshold and 
            not self.is_in_check(board) and
            self.has_non_pawn_material(board)):
            
            # Make null move
            null_board = self.make_null_move(board)
            
            # Search with reduced depth
            null_score = -self.principal_variation_search(
                null_board, depth - self.null_move_reduction - 1, 
                -beta, -beta + 1, False, ply + 1
            )
            
            if null_score >= beta:
                return null_score
        
        # Generate and order moves
        moves = self.get_ordered_moves(board, ply, tt_entry)
        
        if not moves:
            # No legal moves - checkmate or stalemate
            if self.is_in_check(board):
                return -30000 + ply  # Checkmate
            else:
                return 0  # Stalemate
        
        best_score = -float('inf')
        best_move = None
        entry_type = TTEntryType.UPPER_BOUND
        move_count = 0
        
        for move in moves:
            # Check if we should stop
            if self.should_stop_search():
                break
            
            move_count += 1
            
            # Make move
            new_board = self.make_move(board, move)
            
            # Calculate score
            if move_count == 1:
                # Full window search for first move
                score = -self.principal_variation_search(
                    new_board, depth - 1, -beta, -alpha, is_pv_node, ply + 1
                )
            else:
                # Late move reductions
                reduction = 0
                if (depth >= self.lmr_depth_threshold and 
                    move_count >= self.lmr_move_threshold and
                    not is_pv_node and
                    not self.is_tactical_move(move) and
                    not self.is_in_check(new_board)):
                    
                    reduction = max(1, int(self.lmr_reduction_base * math.log(depth) * math.log(move_count)))
                
                # Null window search
                score = -self.principal_variation_search(
                    new_board, depth - 1 - reduction, -alpha - 1, -alpha, False, ply + 1
                )
                
                # Re-search if needed
                if reduction > 0 and score > alpha:
                    score = -self.principal_variation_search(
                        new_board, depth - 1, -alpha - 1, -alpha, False, ply + 1
                    )
                
                # Full window re-search for PV nodes
                if score > alpha and score < beta and is_pv_node:
                    score = -self.principal_variation_search(
                        new_board, depth - 1, -beta, -alpha, True, ply + 1
                    )
            
            # Update best score
            if score > best_score:
                best_score = score
                best_move = move
                
                if score > alpha:
                    alpha = score
                    entry_type = TTEntryType.EXACT
                    
                    # Update principal variation
                    self.pv_table[ply] = [move] + self.pv_table[ply + 1]
                    
                    if score >= beta:
                        # Beta cutoff
                        self.beta_cutoffs += 1
                        entry_type = TTEntryType.LOWER_BOUND
                        
                        # Update move ordering heuristics
                        self.update_move_ordering(move, depth, ply)
                        break
        
        # Store in transposition table
        self.store_transposition_entry(board, depth, best_score, entry_type, best_move)
        
        return best_score
    
    def quiescence_search(self, board, alpha: float, beta: float, ply: int) -> float:
        """
        Quiescence search to resolve tactical sequences
        
        Args:
            board: Current position
            alpha: Alpha bound
            beta: Beta bound
            ply: Current ply
            
        Returns:
            Position evaluation
        """
        self.nodes_searched += 1
        
        # Stand pat evaluation
        stand_pat = self.evaluate_position(board)
        
        if stand_pat >= beta:
            return stand_pat
        
        if stand_pat > alpha:
            alpha = stand_pat
        
        # Generate tactical moves (captures, promotions, checks)
        tactical_moves = self.get_tactical_moves(board)
        
        for move in tactical_moves:
            if self.should_stop_search():
                break
            
            # Delta pruning - skip moves that can't improve alpha
            if self.get_move_material_gain(move) + stand_pat + 200 < alpha:
                continue
            
            # SEE pruning - skip bad captures
            if self.static_exchange_evaluation(board, move) < 0:
                continue
            
            new_board = self.make_move(board, move)
            score = -self.quiescence_search(new_board, -beta, -alpha, ply + 1)
            
            if score >= beta:
                return score
            
            if score > alpha:
                alpha = score
        
        return alpha
    
    def get_ordered_moves(self, board, ply: int, tt_entry: Optional[TranspositionEntry]) -> List[Any]:
        """
        Generate and order moves for optimal alpha-beta performance
        
        Args:
            board: Current position
            ply: Current ply
            tt_entry: Transposition table entry
            
        Returns:
            Ordered list of moves
        """
        moves = self.generate_legal_moves(board)
        if not moves:
            return []
        
        # Move scoring for ordering
        move_scores = []
        
        for move in moves:
            score = 0
            
            # 1. Hash move (from transposition table)
            if tt_entry and tt_entry.best_move == move:
                score += 10000000
            
            # 2. Captures (MVV-LVA)
            if self.is_capture(move):
                victim_value = self.get_piece_value(self.get_captured_piece(board, move))
                attacker_value = self.get_piece_value(self.get_moving_piece(board, move))
                score += 1000000 + victim_value * 10 - attacker_value
            
            # 3. Promotions
            if self.is_promotion(move):
                score += 900000 + self.get_piece_value(self.get_promotion_piece(move))
            
            # 4. Killer moves
            if move in self.killer_moves[ply]:
                score += 800000
            
            # 5. Counter moves
            if ply > 0 and move == self.counter_moves.get(self.get_last_move(board)):
                score += 700000
            
            # 6. History heuristic
            move_key = (self.get_moving_piece(board, move), self.get_move_target(move))
            score += self.history_table[move_key]
            
            # 7. Piece-square table bonus
            score += self.get_piece_square_bonus(move)
            
            move_scores.append((score, move))
        
        # Sort by score (descending)
        move_scores.sort(key=lambda x: x[0], reverse=True)
        return [move for _, move in move_scores]
    
    def update_move_ordering(self, move: Any, depth: int, ply: int):
        """Update move ordering heuristics after beta cutoff"""
        
        # Update killer moves
        if not self.is_capture(move):
            killers = self.killer_moves[ply]
            if move != killers[0]:
                killers[1] = killers[0]
                killers[0] = move
        
        # Update history table
        if not self.is_capture(move):
            move_key = (self.get_moving_piece_at_move(move), self.get_move_target(move))
            self.history_table[move_key] += depth * depth
        
        # Update counter moves
        if ply > 0:
            last_move = self.get_last_move_from_ply(ply - 1)
            if last_move:
                self.counter_moves[last_move] = move
    
    def static_exchange_evaluation(self, board, move) -> int:
        """
        Static Exchange Evaluation - estimate material gain/loss from capture sequence
        
        Args:
            board: Current position
            move: Move to evaluate
            
        Returns:
            Material balance after exchanges
        """
        # Simplified SEE implementation
        if not self.is_capture(move):
            return 0
        
        target_square = self.get_move_target(move)
        captured_value = self.get_piece_value(self.get_captured_piece(board, move))
        attacker_value = self.get_piece_value(self.get_moving_piece(board, move))
        
        # Find all attackers to the target square
        attackers = self.get_attackers_to_square(board, target_square)
        
        if len(attackers) <= 1:
            return captured_value
        
        # Simulate capture sequence
        gain = [captured_value]
        current_attacker_value = attacker_value
        
        for i in range(1, len(attackers)):
            gain.append(current_attacker_value - gain[i-1])
            if max(-gain[i-1], gain[i]) < 0:
                break
            current_attacker_value = self.get_piece_value(attackers[i])
        
        # Work backwards to find the best outcome
        while len(gain) > 1:
            last_gain = gain.pop()
            second_last = gain[-1]
            gain[-1] = max(second_last, -last_gain)
        
        return gain[0] if gain else 0
    
    def should_stop_search(self, time_limit: Optional[float] = None) -> bool:
        """Check if search should be stopped"""
        if self.stop_search:
            return True
        
        if time_limit and time.time() - self.start_time > time_limit:
            return True
        
        return False
    
    def probe_transposition_table(self, board) -> Optional[TranspositionEntry]:
        """Probe transposition table for current position"""
        if not self.zobrist_hasher:
            return None
        
        zobrist_hash = self.zobrist_hasher.hash_position(board)
        return self.transposition_table.get(zobrist_hash)
    
    def store_transposition_entry(self, board, depth: int, evaluation: float, 
                                entry_type: TTEntryType, best_move: Any):
        """Store position in transposition table"""
        if not self.zobrist_hasher:
            return
        
        zobrist_hash = self.zobrist_hasher.hash_position(board)
        
        # Replace if deeper search or same depth with exact score
        existing = self.transposition_table.get(zobrist_hash)
        if existing and existing.depth > depth and existing.entry_type == TTEntryType.EXACT:
            return
        
        entry = TranspositionEntry(
            zobrist_hash=zobrist_hash,
            depth=depth,
            evaluation=evaluation,
            entry_type=entry_type,
            best_move=best_move,
            age=self.tt_age
        )
        
        self.transposition_table[zobrist_hash] = entry
        
        # Garbage collection if table too large
        if len(self.transposition_table) > self.tt_max_size:
            self.cleanup_transposition_table()
    
    def cleanup_transposition_table(self):
        """Clean up transposition table by removing old entries"""
        # Remove entries from previous searches
        to_remove = []
        for key, entry in self.transposition_table.items():
            if entry.age < self.tt_age - 2:  # Keep last 2 search generations
                to_remove.append(key)
        
        for key in to_remove[:len(to_remove)//2]:  # Remove half of old entries
            del self.transposition_table[key]
    
    # Abstract methods to be implemented by specific chess engine
    def generate_legal_moves(self, board) -> List[Any]:
        """Generate all legal moves"""
        if self.move_generator:
            return self.move_generator.generate_legal_moves(board)
        return []
    
    def make_move(self, board, move):
        """Make a move and return new board state"""
        # Should be implemented by specific chess implementation
        return board
    
    def make_null_move(self, board):
        """Make a null move (pass turn)"""
        # Should be implemented by specific chess implementation
        return board
    
    def evaluate_position(self, board) -> float:
        """Evaluate board position"""
        if self.evaluator:
            return self.evaluator.evaluate(board)
        return 0.0
    
    def is_in_check(self, board) -> bool:
        """Check if current player is in check"""
        return False  # Implement in subclass
    
    def has_non_pawn_material(self, board) -> bool:
        """Check if current player has non-pawn material"""
        return True  # Implement in subclass
    
    def is_capture(self, move) -> bool:
        """Check if move is a capture"""
        return False  # Implement in subclass
    
    def is_promotion(self, move) -> bool:
        """Check if move is a promotion"""
        return False  # Implement in subclass
    
    def is_tactical_move(self, move) -> bool:
        """Check if move is tactical (capture, promotion, check)"""
        return self.is_capture(move) or self.is_promotion(move)
    
    def get_tactical_moves(self, board) -> List[Any]:
        """Get tactical moves (captures, promotions, checks)"""
        moves = self.generate_legal_moves(board)
        return [move for move in moves if self.is_tactical_move(move)]
    
    def get_piece_value(self, piece) -> int:
        """Get material value of piece"""
        values = {'p': 100, 'n': 300, 'b': 300, 'r': 500, 'q': 900, 'k': 10000}
        return values.get(piece.lower() if piece else '', 0)
    
    def get_captured_piece(self, board, move):
        """Get piece captured by move"""
        return None  # Implement in subclass
    
    def get_moving_piece(self, board, move):
        """Get piece making the move"""
        return None  # Implement in subclass
    
    def get_move_target(self, move):
        """Get target square of move"""
        return None  # Implement in subclass
    
    def get_attackers_to_square(self, board, square) -> List[Any]:
        """Get all pieces attacking a square"""
        return []  # Implement in subclass
    
    def get_piece_square_bonus(self, move) -> int:
        """Get piece-square table bonus for move"""
        return 0  # Implement in subclass
    
    def get_move_material_gain(self, move) -> int:
        """Get material gain from move"""
        return 0  # Implement in subclass
