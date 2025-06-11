#!/usr/bin/env python3
"""
Advanced Chess Search Engine with Neural Network Integration
===========================================================
Implements sophisticated search algorithms including MCTS and enhanced Alpha-Beta
with neural network evaluation following world-class AI development principles.
"""

import math
import time
import random
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from collections import defaultdict
import torch
import torch.nn.functional as F

from ..core.board import ChessBoard
from ..core.moves import MoveGenerator, Move
from ..core.evaluation import Evaluator
from .network import AlphaZeroNetwork


@dataclass
class SearchStats:
    """Statistics for search performance analysis."""
    nodes_searched: int = 0
    time_elapsed: float = 0.0
    depth_reached: int = 0
    evaluations: int = 0
    cache_hits: int = 0
    principal_variation: List[Move] = None
    best_score: float = 0.0


class TranspositionTable:
    """High-performance transposition table with Zobrist hashing."""
    
    def __init__(self, size_mb: int = 256):
        self.size = (size_mb * 1024 * 1024) // 64  # Approximate entries
        self.table = {}
        self.hits = 0
        self.misses = 0
    
    def store(self, zobrist_hash: int, depth: int, score: float, 
              move: Optional[Move], node_type: str):
        """Store position evaluation in transposition table."""
        self.table[zobrist_hash] = {
            'depth': depth,
            'score': score,
            'move': move,
            'type': node_type,  # 'exact', 'lower', 'upper'
            'timestamp': time.time()
        }
    
    def probe(self, zobrist_hash: int, depth: int, alpha: float, beta: float):
        """Probe transposition table for stored evaluation."""
        if zobrist_hash in self.table:
            entry = self.table[zobrist_hash]
            if entry['depth'] >= depth:
                self.hits += 1
                score = entry['score']
                
                if entry['type'] == 'exact':
                    return score, entry['move']
                elif entry['type'] == 'lower' and score >= beta:
                    return score, entry['move']
                elif entry['type'] == 'upper' and score <= alpha:
                    return score, entry['move']
            
            return None, entry.get('move')
        
        self.misses += 1
        return None, None
    
    def clear(self):
        """Clear the transposition table."""
        self.table.clear()
        self.hits = 0
        self.misses = 0


class MCTSNode:
    """Monte Carlo Tree Search node for neural network guided search."""
    
    def __init__(self, board: ChessBoard, parent=None, move: Move = None, prior: float = 0.0):
        self.board = board
        self.parent = parent
        self.move = move
        self.prior = prior
        
        # MCTS statistics
        self.visits = 0
        self.value_sum = 0.0
        self.children = {}
        self.expanded = False
        
        # Virtual loss for parallel search
        self.virtual_loss = 0
    
    def is_expanded(self) -> bool:
        return self.expanded
    
    def value(self) -> float:
        """Average value of this node."""
        if self.visits == 0:
            return 0.0
        return (self.value_sum - self.virtual_loss) / (self.visits + self.virtual_loss)
    
    def ucb_score(self, c_puct: float = 1.0) -> float:
        """Upper Confidence Bound score for node selection."""
        if self.visits == 0:
            return float('inf')
        
        exploration = c_puct * self.prior * math.sqrt(self.parent.visits) / (1 + self.visits)
        return self.value() + exploration
    
    def select_child(self, c_puct: float = 1.0):
        """Select best child using UCB."""
        return max(self.children.values(), key=lambda node: node.ucb_score(c_puct))
    
    def expand(self, priors: Dict[Move, float]):
        """Expand node with child nodes."""
        for move, prior in priors.items():
            new_board = self.board.copy()
            new_board.make_move(move)
            self.children[move] = MCTSNode(new_board, parent=self, move=move, prior=prior)
        self.expanded = True
    
    def backup(self, value: float):
        """Backup value through the tree."""
        self.visits += 1
        self.value_sum += value
        if self.parent:
            self.parent.backup(-value)  # Flip value for opponent
    
    def add_virtual_loss(self):
        """Add virtual loss for parallel search."""
        self.virtual_loss += 1
    
    def remove_virtual_loss(self):
        """Remove virtual loss after search completion."""
        self.virtual_loss = max(0, self.virtual_loss - 1)


class AdvancedSearchEngine:
    """
    Advanced search engine implementing both MCTS and enhanced Alpha-Beta search
    with neural network evaluation, following world-class AI development principles.
    """
    
    def __init__(self, network: Optional[AlphaZeroNetwork] = None, 
                 evaluator: Optional[Evaluator] = None,
                 move_generator: Optional[MoveGenerator] = None):
        self.network = network
        self.evaluator = evaluator or Evaluator()
        self.move_generator = move_generator or MoveGenerator()
        
        # Search configuration
        self.use_neural_network = network is not None
        self.transposition_table = TranspositionTable(size_mb=512)
        
        # Search statistics
        self.stats = SearchStats()
        
        # MCTS configuration
        self.mcts_simulations = 800
        self.c_puct = 1.0
        self.dirichlet_alpha = 0.3
        self.dirichlet_epsilon = 0.25
        
        # Alpha-Beta configuration
        self.killer_moves = defaultdict(list)  # [depth][move]
        self.history_heuristic = defaultdict(int)
        self.max_quiescence_depth = 8
    
    def search(self, board: ChessBoard, depth: int = 6, time_limit: float = 5.0) -> Tuple[Move, float]:
        """
        Main search entry point - chooses between MCTS and Alpha-Beta based on configuration.
        """
        start_time = time.time()
        self.stats = SearchStats()
        
        if self.use_neural_network and self.network:
            return self.mcts_search(board, time_limit)
        else:
            return self.alpha_beta_search(board, depth, time_limit)
    
    def mcts_search(self, board: ChessBoard, time_limit: float) -> Tuple[Move, float]:
        """
        Monte Carlo Tree Search guided by neural network evaluation.
        Implements advanced MCTS with virtual loss and Dirichlet noise.
        """
        root = MCTSNode(board.copy())
        start_time = time.time()
        
        # Add Dirichlet noise to root priors for exploration
        if self.network:
            priors, value = self.network.evaluate(board)
            if self.dirichlet_epsilon > 0:
                noise = np.random.dirichlet([self.dirichlet_alpha] * len(priors))
                for i, (move, prior) in enumerate(priors.items()):
                    priors[move] = (1 - self.dirichlet_epsilon) * prior + self.dirichlet_epsilon * noise[i]
            root.expand(priors)
        
        simulations = 0
        while time.time() - start_time < time_limit and simulations < self.mcts_simulations:
            self.mcts_simulation(root)
            simulations += 1
        
        # Select best move based on visit count
        if not root.children:
            legal_moves = self.move_generator.generate_legal_moves(board)
            return legal_moves[0] if legal_moves else None, 0.0
        
        best_move = max(root.children.keys(), key=lambda move: root.children[move].visits)
        best_value = root.children[best_move].value()
        
        self.stats.nodes_searched = simulations
        self.stats.time_elapsed = time.time() - start_time
        self.stats.best_score = best_value
        
        return best_move, best_value
    
    def mcts_simulation(self, root: MCTSNode):
        """Single MCTS simulation with neural network guidance."""
        path = []
        node = root
        
        # Selection phase - traverse to leaf
        while node.is_expanded() and node.children:
            node.add_virtual_loss()
            path.append(node)
            node = node.select_child(self.c_puct)
        
        # Expansion and evaluation
        if not node.is_expanded():
            if self.network:
                priors, value = self.network.evaluate(node.board)
                node.expand(priors)
            else:
                # Fallback to classical evaluation
                legal_moves = self.move_generator.generate_legal_moves(node.board)
                uniform_prior = 1.0 / len(legal_moves) if legal_moves else 1.0
                priors = {move: uniform_prior for move in legal_moves}
                node.expand(priors)
                value = self.evaluator.evaluate(node.board)
        else:
            value = node.value()
        
        # Backup phase
        node.backup(value)
        
        # Remove virtual losses
        for path_node in path:
            path_node.remove_virtual_loss()
    
    def alpha_beta_search(self, board: ChessBoard, depth: int, time_limit: float) -> Tuple[Move, float]:
        """
        Enhanced Alpha-Beta search with advanced pruning and move ordering.
        Implements iterative deepening with sophisticated heuristics.
        """
        start_time = time.time()
        best_move = None
        best_score = -float('inf')
        
        # Iterative deepening
        for current_depth in range(1, depth + 1):
            if time.time() - start_time > time_limit * 0.8:
                break
            
            score, move = self.negamax(board, current_depth, -float('inf'), float('inf'), 
                                     board.current_player == 'white', start_time, time_limit)
            
            if move:
                best_move = move
                best_score = score
                self.stats.depth_reached = current_depth
        
        self.stats.time_elapsed = time.time() - start_time
        return best_move, best_score
    
    def negamax(self, board: ChessBoard, depth: int, alpha: float, beta: float, 
                maximizing: bool, start_time: float, time_limit: float) -> Tuple[float, Move]:
        """
        Negamax search with alpha-beta pruning and advanced optimizations.
        """
        if time.time() - start_time > time_limit:
            return 0.0, None
        
        self.stats.nodes_searched += 1
        
        # Transposition table probe
        zobrist_hash = board.get_zobrist_hash() if hasattr(board, 'get_zobrist_hash') else hash(str(board))
        tt_score, tt_move = self.transposition_table.probe(zobrist_hash, depth, alpha, beta)
        if tt_score is not None:
            return tt_score, tt_move
        
        # Terminal conditions
        if depth == 0:
            return self.quiescence_search(board, alpha, beta, self.max_quiescence_depth), None
        
        legal_moves = self.move_generator.generate_legal_moves(board)
        if not legal_moves:
            if board.is_in_check(board.current_player):
                return -10000 + (10 - depth), None  # Checkmate
            return 0, None  # Stalemate
        
        # Move ordering for better alpha-beta pruning
        legal_moves = self.order_moves(board, legal_moves, tt_move, depth)
        
        best_score = -float('inf')
        best_move = None
        original_alpha = alpha
        
        for i, move in enumerate(legal_moves):
            # Make move
            captured = board.make_move(move)
            
            # Late move reductions
            reduction = 0
            if i > 3 and depth > 2 and not move.is_capture() and not board.is_in_check(board.current_player):
                reduction = 1
            
            # Recursive search
            score, _ = self.negamax(board, depth - 1 - reduction, -beta, -alpha, not maximizing, start_time, time_limit)
            score = -score
            
            # Unmake move
            board.unmake_move(move, captured)
            
            if score > best_score:
                best_score = score
                best_move = move
            
            alpha = max(alpha, score)
            if alpha >= beta:
                # Update killer moves and history heuristic
                if not move.is_capture():
                    self.killer_moves[depth].append(move)
                    if len(self.killer_moves[depth]) > 2:
                        self.killer_moves[depth].pop(0)
                    self.history_heuristic[move] += depth * depth
                break
        
        # Store in transposition table
        node_type = 'exact' if best_score > original_alpha and best_score < beta else 'lower' if best_score >= beta else 'upper'
        self.transposition_table.store(zobrist_hash, depth, best_score, best_move, node_type)
        
        return best_score, best_move
    
    def quiescence_search(self, board: ChessBoard, alpha: float, beta: float, depth: int) -> float:
        """
        Quiescence search to avoid horizon effect.
        Searches only tactical moves (captures, checks, promotions).
        """
        if depth <= 0:
            return self.evaluate_position(board)
        
        stand_pat = self.evaluate_position(board)
        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat
        
        # Generate only tactical moves
        tactical_moves = self.move_generator.generate_tactical_moves(board)
        tactical_moves = sorted(tactical_moves, key=lambda m: self.move_value(board, m), reverse=True)
        
        for move in tactical_moves:
            if self.see_score(board, move) < 0:  # Static Exchange Evaluation
                continue
            
            captured = board.make_move(move)
            score = -self.quiescence_search(board, -beta, -alpha, depth - 1)
            board.unmake_move(move, captured)
            
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        
        return alpha
    
    def order_moves(self, board: ChessBoard, moves: List[Move], hash_move: Move, depth: int) -> List[Move]:
        """
        Sophisticated move ordering for optimal alpha-beta pruning.
        """
        def move_priority(move):
            score = 0
            
            # Hash move gets highest priority
            if move == hash_move:
                score += 10000
            
            # Captures ordered by MVV-LVA (Most Valuable Victim - Least Valuable Attacker)
            if move.is_capture():
                victim_value = self.piece_value(board.get_piece_at(move.to_square))
                attacker_value = self.piece_value(board.get_piece_at(move.from_square))
                score += 1000 + victim_value - attacker_value
            
            # Killer moves
            if move in self.killer_moves[depth]:
                score += 900
            
            # History heuristic
            score += self.history_heuristic.get(move, 0)
            
            # Promotions
            if move.is_promotion():
                score += 800
            
            # Checks
            if self.gives_check(board, move):
                score += 100
            
            return score
        
        return sorted(moves, key=move_priority, reverse=True)
    
    def evaluate_position(self, board: ChessBoard) -> float:
        """Evaluate position using neural network or classical evaluation."""
        self.stats.evaluations += 1
        
        if self.use_neural_network and self.network:
            _, value = self.network.evaluate(board)
            return value
        else:
            return self.evaluator.evaluate(board)
    
    def see_score(self, board: ChessBoard, move: Move) -> float:
        """Static Exchange Evaluation for tactical moves."""
        # Simplified SEE implementation
        # In a full implementation, this would simulate the exchange sequence
        if not move.is_capture():
            return 0.0
        
        victim_value = self.piece_value(board.get_piece_at(move.to_square))
        attacker_value = self.piece_value(board.get_piece_at(move.from_square))
        
        return victim_value - attacker_value
    
    def piece_value(self, piece) -> int:
        """Get piece value for move ordering."""
        if not piece:
            return 0
        
        values = {'p': 100, 'n': 300, 'b': 300, 'r': 500, 'q': 900, 'k': 10000}
        return values.get(piece.lower(), 0)
    
    def move_value(self, board: ChessBoard, move: Move) -> int:
        """Calculate move value for ordering."""
        value = 0
        if move.is_capture():
            value += self.piece_value(board.get_piece_at(move.to_square)) * 10
        if move.is_promotion():
            value += 800
        return value
    
    def gives_check(self, board: ChessBoard, move: Move) -> bool:
        """Check if move gives check to opponent."""
        # Simplified check detection
        captured = board.make_move(move)
        gives_check = board.is_in_check(board.get_opponent_color())
        board.unmake_move(move, captured)
        return gives_check
    
    def get_search_info(self) -> Dict[str, Any]:
        """Get detailed search information for analysis."""
        return {
            'nodes': self.stats.nodes_searched,
            'time': self.stats.time_elapsed,
            'depth': self.stats.depth_reached,
            'evaluations': self.stats.evaluations,
            'nps': self.stats.nodes_searched / max(self.stats.time_elapsed, 0.001),
            'tt_hits': self.transposition_table.hits,
            'tt_misses': self.transposition_table.misses,
            'best_score': self.stats.best_score
        }
