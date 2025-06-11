"""
Monte Carlo Tree Search (MCTS) Implementation
============================================

Advanced MCTS implementation guided by neural network for the grandmaster-level chess engine.
Implements UCT algorithm with neural network policy and value guidance.
"""

import math
import random
import time
import threading
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from core.board import ChessBoard, Move, Color
from core.moves import MoveGenerator
from neural.network import NeuralNetworkEvaluator


@dataclass
class MCTSConfig:
    """Configuration parameters for MCTS."""
    # UCT parameters
    c_puct: float = 1.0          # Exploration constant
    c_base: float = 19652        # Base for UCB formula
    c_init: float = 1.25         # Initial UCB value
    
    # Search limits
    max_simulations: int = 800   # Number of MCTS simulations
    max_time: float = 10.0       # Maximum search time in seconds
    max_depth: int = 100         # Maximum search depth
    
    # Evaluation parameters
    fpu_reduction: float = 0.2   # First Play Urgency reduction
    virtual_loss: int = 3        # Virtual loss for parallelization
    
    # Noise parameters (for self-play)
    add_dirichlet_noise: bool = False
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    
    # Temperature for move selection
    temperature: float = 1.0


@dataclass
class SearchStats:
    """Statistics from MCTS search."""
    simulations: int = 0
    time_used: float = 0.0
    nodes_created: int = 0
    cache_hits: int = 0
    nps: float = 0.0  # Nodes per second
    
    def update_nps(self):
        """Update nodes per second calculation."""
        if self.time_used > 0:
            self.nps = self.simulations / self.time_used


class MCTSNode:
    """
    A node in the MCTS tree.
    
    Represents a board position and maintains statistics for the UCT algorithm.
    """
    
    def __init__(self, board: ChessBoard, parent: Optional['MCTSNode'] = None, 
                 prior_prob: float = 0.0, move: Optional[Move] = None):
        # Tree structure
        self.board = board.copy()
        self.parent = parent
        self.children: Dict[str, 'MCTSNode'] = {}
        self.move = move  # Move that led to this position
        
        # MCTS statistics
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior_prob = prior_prob
        self.virtual_loss = 0
        
        # Cached data
        self.legal_moves: Optional[List[Move]] = None
        self.is_terminal: Optional[bool] = None
        self.terminal_value: Optional[float] = None
        
        # Thread safety
        self.lock = threading.Lock()
    
    def is_leaf(self) -> bool:
        """Check if this is a leaf node (no children expanded)."""
        return len(self.children) == 0
    
    def is_fully_expanded(self) -> bool:
        """Check if all legal moves have been expanded."""
        if self.legal_moves is None:
            return False
        return len(self.children) == len(self.legal_moves)
    
    def get_q_value(self) -> float:
        """Get the average value (Q-value) of this node."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def get_ucb_score(self, c_puct: float, parent_visits: int) -> float:
        """
        Calculate UCB (Upper Confidence Bound) score for this node.
        
        Uses the UCT formula: Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        """
        if self.visit_count == 0:
            # First Play Urgency - high value to encourage exploration
            return float('inf')
        
        # Q-value
        q_value = self.get_q_value()
        
        # UCB term
        ucb_term = c_puct * self.prior_prob * math.sqrt(parent_visits) / (1 + self.visit_count)
        
        return q_value + ucb_term
    
    def select_child(self, c_puct: float) -> 'MCTSNode':
        """Select the child with the highest UCB score."""
        if not self.children:
            raise ValueError("Cannot select child from node with no children")
        
        best_score = float('-inf')
        best_child = None
        
        for child in self.children.values():
            score = child.get_ucb_score(c_puct, self.visit_count)
            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child
    
    def add_child(self, move: Move, prior_prob: float) -> 'MCTSNode':
        """Add a child node for the given move."""
        board_copy = self.board.copy()
        board_copy.make_move(move)
        
        move_key = str(move)
        child = MCTSNode(board_copy, parent=self, prior_prob=prior_prob, move=move)
        self.children[move_key] = child
        
        return child
    
    def backup(self, value: float):
        """
        Backup the value through the tree to the root.
        
        Updates visit counts and value sums for all ancestors.
        """
        current = self
        while current is not None:
            with current.lock:
                current.visit_count += 1
                current.value_sum += value
            
            # Flip value for opponent's perspective
            value = -value
            current = current.parent
    
    def get_child_visit_counts(self) -> Dict[str, int]:
        """Get visit counts for all children."""
        return {move_key: child.visit_count for move_key, child in self.children.items()}
    
    def get_best_move(self, temperature: float = 0.0) -> Optional[Move]:
        """
        Get the best move based on visit counts.
        
        Args:
            temperature: Controls randomness in move selection
                        0.0 = always select most visited move
                        >0.0 = probabilistic selection based on visit counts
        """
        if not self.children:
            return None
        
        if temperature == 0.0:
            # Deterministic selection - most visited move
            best_child = max(self.children.values(), key=lambda c: c.visit_count)
            return best_child.move
        else:
            # Probabilistic selection with temperature
            visit_counts = np.array([child.visit_count for child in self.children.values()])
            moves = [child.move for child in self.children.values()]
            
            # Apply temperature
            if temperature != 1.0:
                visit_counts = visit_counts ** (1.0 / temperature)
            
            # Normalize to probabilities
            probs = visit_counts / visit_counts.sum()
            
            # Sample move
            move_idx = np.random.choice(len(moves), p=probs)
            return moves[move_idx]


class MCTS:
    """
    Monte Carlo Tree Search implementation for chess.
    
    Uses neural network guidance for move probabilities and position evaluation.
    Implements the UCT algorithm with various enhancements for chess.
    """
    
    def __init__(self, neural_net: NeuralNetworkEvaluator, move_generator: MoveGenerator,
                 config: MCTSConfig = None):
        self.neural_net = neural_net
        self.move_generator = move_generator
        self.config = config or MCTSConfig()
        
        # Search state
        self.root: Optional[MCTSNode] = None
        self.stats = SearchStats()
        
        # Caching for neural network calls
        self.evaluation_cache: Dict[str, Tuple[float, Dict[str, float]]] = {}
        self.cache_max_size = 10000
    
    def search(self, board: ChessBoard, max_simulations: Optional[int] = None,
               max_time: Optional[float] = None) -> Tuple[Move, SearchStats]:
        """
        Perform MCTS search to find the best move.
        
        Args:
            board: Current board position
            max_simulations: Maximum number of simulations (overrides config)
            max_time: Maximum search time in seconds (overrides config)
            
        Returns:
            Tuple of (best_move, search_statistics)
        """
        # Initialize search
        self.root = MCTSNode(board)
        self.stats = SearchStats()
        
        # Search limits
        sim_limit = max_simulations or self.config.max_simulations
        time_limit = max_time or self.config.max_time
        
        start_time = time.time()
        
        # Main search loop
        while (self.stats.simulations < sim_limit and 
               time.time() - start_time < time_limit):
            
            # Single MCTS simulation
            self._simulate()
            self.stats.simulations += 1
        
        # Update statistics
        self.stats.time_used = time.time() - start_time
        self.stats.update_nps()
        
        # Select best move
        best_move = self.root.get_best_move(temperature=0.0)
        
        return best_move, self.stats
    
    def _simulate(self):
        """Perform a single MCTS simulation."""
        # Selection phase: traverse tree to leaf node
        leaf = self._select()
        
        # Expansion and evaluation
        if self._is_terminal(leaf):
            # Terminal node - use exact value
            value = self._get_terminal_value(leaf)
        else:
            # Expand and evaluate with neural network
            value = self._expand_and_evaluate(leaf)
        
        # Backup phase: propagate value up the tree
        leaf.backup(value)
    
    def _select(self) -> MCTSNode:
        """
        Selection phase: traverse tree using UCT until reaching a leaf node.
        
        Returns:
            Leaf node for expansion/evaluation
        """
        current = self.root
        
        while not current.is_leaf() and not self._is_terminal(current):
            current = current.select_child(self.config.c_puct)
        
        return current
    
    def _expand_and_evaluate(self, node: MCTSNode) -> float:
        """
        Expansion and evaluation phase.
        
        Expands the node (adds children) and evaluates with neural network.
        
        Args:
            node: Node to expand and evaluate
            
        Returns:
            Evaluation value from the neural network
        """
        # Get neural network evaluation and move probabilities
        board_key = node.board.to_fen()
        
        if board_key in self.evaluation_cache:
            value, move_probs = self.evaluation_cache[board_key]
            self.stats.cache_hits += 1
        else:
            # Neural network evaluation
            value = self.neural_net.evaluate_position(node.board)
            move_probs = self.neural_net.get_move_probabilities(node.board)
            
            # Cache the result
            if len(self.evaluation_cache) < self.cache_max_size:
                self.evaluation_cache[board_key] = (value, move_probs)
        
        # Expand node with legal moves
        legal_moves = self.move_generator.generate_legal_moves(node.board)
        node.legal_moves = legal_moves
        
        # Add children for legal moves
        total_prior = 0.0
        for move in legal_moves:
            move_key = str(move)
            # Get prior probability from neural network
            prior_prob = move_probs.get(move.to_square, 0.001)  # Default small prob
            total_prior += prior_prob
        
        # Normalize priors if necessary
        if total_prior > 0:
            for move in legal_moves:
                move_key = str(move)
                prior_prob = move_probs.get(move.to_square, 0.001) / total_prior
                
                # Add Dirichlet noise for exploration during self-play
                if self.config.add_dirichlet_noise and node == self.root:
                    noise = np.random.gamma(self.config.dirichlet_alpha)
                    prior_prob = ((1 - self.config.dirichlet_epsilon) * prior_prob + 
                                 self.config.dirichlet_epsilon * noise)
                
                node.add_child(move, prior_prob)
        
        self.stats.nodes_created += len(legal_moves)
        
        # Adjust value for current player
        if node.board.to_move == Color.BLACK:
            value = -value
        
        return value
    
    def _is_terminal(self, node: MCTSNode) -> bool:
        """Check if the node represents a terminal position."""
        if node.is_terminal is not None:
            return node.is_terminal
        
        # Check for checkmate or stalemate
        legal_moves = self.move_generator.generate_legal_moves(node.board)
        
        if len(legal_moves) == 0:
            # No legal moves - either checkmate or stalemate
            node.is_terminal = True
            if self.move_generator.is_in_check(node.board, node.board.to_move):
                # Checkmate
                node.terminal_value = -1.0  # Loss for current player
            else:
                # Stalemate
                node.terminal_value = 0.0   # Draw
        else:
            # Check for other draw conditions
            if (node.board.halfmove_clock >= 100 or  # 50-move rule
                node.board.is_threefold_repetition()):
                node.is_terminal = True
                node.terminal_value = 0.0
            else:
                node.is_terminal = False
        
        return node.is_terminal
    
    def _get_terminal_value(self, node: MCTSNode) -> float:
        """Get the exact value for a terminal position."""
        if node.terminal_value is not None:
            return node.terminal_value
        
        # Should not reach here if _is_terminal was called first
        return 0.0
    
    def get_principal_variation(self, max_depth: int = 10) -> List[Move]:
        """
        Get the principal variation (best line) from the search tree.
        
        Args:
            max_depth: Maximum depth to traverse
            
        Returns:
            List of moves in the principal variation
        """
        pv = []
        current = self.root
        
        for _ in range(max_depth):
            if not current or not current.children:
                break
            
            # Find most visited child
            best_child = max(current.children.values(), key=lambda c: c.visit_count)
            pv.append(best_child.move)
            current = best_child
        
        return pv
    
    def get_move_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get detailed statistics for all root moves.
        
        Returns:
            Dictionary mapping move strings to statistics
        """
        if not self.root or not self.root.children:
            return {}
        
        stats = {}
        total_visits = self.root.visit_count
        
        for move_key, child in self.root.children.items():
            if child.visit_count > 0:
                stats[move_key] = {
                    'visits': child.visit_count,
                    'visit_percentage': 100.0 * child.visit_count / total_visits,
                    'q_value': child.get_q_value(),
                    'prior_prob': child.prior_prob,
                    'ucb_score': child.get_ucb_score(self.config.c_puct, total_visits)
                }
        
        return stats
    
    def clear_cache(self):
        """Clear the evaluation cache."""
        self.evaluation_cache.clear()


class MCTSSearchEngine:
    """
    High-level interface for MCTS-based chess engine.
    
    Integrates MCTS with neural network evaluation and provides
    a clean interface for the UCI engine.
    """
    
    def __init__(self, neural_net: NeuralNetworkEvaluator, move_generator: MoveGenerator):
        self.neural_net = neural_net
        self.move_generator = move_generator
        self.mcts = MCTS(neural_net, move_generator)
        
        # Search configuration
        self.default_config = MCTSConfig()
    
    def search_position(self, board: ChessBoard, time_limit: float = 10.0,
                       simulations: int = 800) -> Tuple[Move, Dict]:
        """
        Search for the best move in the given position.
        
        Args:
            board: Current board position
            time_limit: Maximum search time in seconds
            simulations: Number of MCTS simulations
            
        Returns:
            Tuple of (best_move, search_info)
        """
        # Configure search
        config = MCTSConfig()
        config.max_time = time_limit
        config.max_simulations = simulations
        
        # Create new MCTS instance for this search
        mcts = MCTS(self.neural_net, self.move_generator, config)
        
        # Perform search
        best_move, stats = mcts.search(board)
        
        # Prepare search info
        search_info = {
            'best_move': best_move,
            'simulations': stats.simulations,
            'time': stats.time_used,
            'nps': stats.nps,
            'nodes_created': stats.nodes_created,
            'cache_hits': stats.cache_hits,
            'principal_variation': mcts.get_principal_variation(),
            'move_stats': mcts.get_move_statistics()
        }
        
        return best_move, search_info
    
    def analyze_position(self, board: ChessBoard, depth: int = 5) -> Dict:
        """
        Analyze a position and return detailed evaluation.
        
        Args:
            board: Position to analyze
            depth: Analysis depth
            
        Returns:
            Analysis results
        """
        # Quick evaluation with neural network
        position_value = self.neural_net.evaluate_position(board)
        move_probs = self.neural_net.get_move_probabilities(board)
        
        # Get legal moves with their probabilities
        legal_moves = self.move_generator.generate_legal_moves(board)
        move_analysis = []
        
        for move in legal_moves[:10]:  # Top 10 moves
            prob = move_probs.get(move.to_square, 0.0)
            move_analysis.append({
                'move': move,
                'probability': prob,
                'move_str': str(move)
            })
        
        # Sort by probability
        move_analysis.sort(key=lambda x: x['probability'], reverse=True)
        
        return {
            'position_value': position_value,
            'best_moves': move_analysis,
            'legal_move_count': len(legal_moves)
        }


# Example usage and testing
if __name__ == "__main__":
    print("Testing MCTS implementation...")
    
    # Import required modules
    from core.board import ChessBoard
    from core.moves import MoveGenerator
    from neural.network import create_default_network
    
    # Initialize components
    board = ChessBoard()
    move_gen = MoveGenerator()
    neural_net = create_default_network()
    
    # Create MCTS engine
    mcts_engine = MCTSSearchEngine(neural_net, move_gen)
    
    # Test position analysis
    print("Analyzing starting position...")
    analysis = mcts_engine.analyze_position(board)
    print(f"Position value: {analysis['position_value']:.3f}")
    print(f"Legal moves: {analysis['legal_move_count']}")
    print("Top moves:")
    for i, move_info in enumerate(analysis['best_moves'][:5]):
        print(f"  {i+1}. {move_info['move_str']} (prob: {move_info['probability']:.3f})")
    
    # Test short search
    print("\nPerforming short MCTS search...")
    best_move, search_info = mcts_engine.search_position(board, time_limit=2.0, simulations=100)
    
    if best_move:
        print(f"Best move: {best_move}")
        print(f"Simulations: {search_info['simulations']}")
        print(f"Time: {search_info['time']:.2f}s")
        print(f"Nodes/sec: {search_info['nps']:.0f}")
        print(f"Principal variation: {[str(m) for m in search_info['principal_variation'][:3]]}")
    else:
        print("No move found!")
    
    print("MCTS tests completed!")

# Alias for backward compatibility
MCTSEngine = MCTSSearchEngine
