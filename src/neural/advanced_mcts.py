#!/usr/bin/env python3
"""
Advanced Monte Carlo Tree Search Implementation
=============================================
AlphaZero-style MCTS with neural network guidance, featuring:
- PUCT exploration formula
- Virtual loss for parallel search
- Dirichlet noise for exploration
- Temperature scaling
- Transposition tables for MCTS nodes
"""

import math
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict
import threading
import time

@dataclass
class MCTSNode:
    """MCTS tree node with neural network priors and statistics"""
    visit_count: int = 0
    value_sum: float = 0.0
    prior: float = 0.0
    children: Dict[Any, 'MCTSNode'] = None
    virtual_loss: int = 0
    state_hash: Optional[int] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = {}
    
    @property
    def value(self) -> float:
        """Average value from visits"""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    @property
    def adjusted_visit_count(self) -> int:
        """Visit count adjusted for virtual loss"""
        return max(1, self.visit_count + self.virtual_loss)
    
    def is_expanded(self) -> bool:
        """Check if node has been expanded"""
        return len(self.children) > 0
    
    def select_child(self, c_puct: float = 1.0) -> Tuple[Any, 'MCTSNode']:
        """Select child using PUCT formula"""
        best_score = -float('inf')
        best_action = None
        best_child = None
        
        sqrt_parent_visits = math.sqrt(self.adjusted_visit_count)
        
        for action, child in self.children.items():
            # PUCT formula: Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
            q_value = child.value
            u_value = c_puct * child.prior * sqrt_parent_visits / child.adjusted_visit_count
            score = q_value + u_value
            
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        
        return best_action, best_child

class AdvancedMCTS:
    """
    Advanced MCTS implementation with neural network guidance
    
    Features:
    - Policy-Value network integration
    - Virtual loss for parallel search
    - Dirichlet noise for exploration
    - Temperature-based action selection
    - Transposition table for node reuse
    """
    
    def __init__(self, 
                 neural_network=None,
                 c_puct: float = 1.0,
                 dirichlet_alpha: float = 0.3,
                 dirichlet_epsilon: float = 0.25,
                 temperature: float = 1.0,
                 use_transposition_table: bool = True,
                 max_simulations: int = 800):
        
        self.neural_network = neural_network
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.temperature = temperature
        self.use_transposition_table = use_transposition_table
        self.max_simulations = max_simulations
        
        # Transposition table for MCTS nodes
        self.transposition_table: Dict[int, MCTSNode] = {}
        self.lock = threading.Lock()
        
        # Search statistics
        self.nodes_searched = 0
        self.search_time = 0.0
        
    def search(self, board, simulations: int = None) -> Tuple[Any, float]:
        """
        Perform MCTS search and return best move with value
        
        Args:
            board: Chess board state
            simulations: Number of simulations to run
            
        Returns:
            Tuple of (best_action, estimated_value)
        """
        start_time = time.time()
        simulations = simulations or self.max_simulations
        
        # Get or create root node
        root_hash = hash(str(board)) if hasattr(board, '__str__') else 0
        root = self.get_or_create_node(root_hash)
        
        # Expand root if needed
        if not root.is_expanded():
            self.expand_node(root, board)
        
        # Add Dirichlet noise to root for exploration
        self.add_dirichlet_noise(root)
        
        # Run simulations
        for _ in range(simulations):
            self.simulate(root, board.copy() if hasattr(board, 'copy') else board)
            self.nodes_searched += 1
        
        self.search_time = time.time() - start_time
        
        # Select best action based on visit counts
        return self.select_action(root)
    
    def simulate(self, node: MCTSNode, board) -> float:
        """
        Run single MCTS simulation from given node
        
        Args:
            node: Current MCTS node
            board: Current board state
            
        Returns:
            Value estimate from simulation
        """
        path = []
        current_node = node
        current_board = board
        
        # Selection phase - traverse down the tree
        while current_node.is_expanded() and not self.is_terminal(current_board):
            action, current_node = current_node.select_child(self.c_puct)
            path.append((current_node, action))
            
            # Apply virtual loss
            with self.lock:
                current_node.virtual_loss += 1
            
            # Update board state
            current_board = self.apply_action(current_board, action)
        
        # Expansion and evaluation
        value = 0.0
        if not self.is_terminal(current_board):
            # Expand node if not terminal
            if not current_node.is_expanded():
                self.expand_node(current_node, current_board)
            
            # Evaluate using neural network or rollout
            value = self.evaluate_position(current_board)
        else:
            # Terminal position - get game result
            value = self.get_game_result(current_board)
        
        # Backup phase - propagate value up the tree
        self.backup(path, value)
        
        return value
    
    def expand_node(self, node: MCTSNode, board):
        """
        Expand MCTS node using neural network priors
        
        Args:
            node: Node to expand
            board: Board state at this node
        """
        legal_actions = self.get_legal_actions(board)
        
        if not legal_actions:
            return
        
        # Get neural network priors and value
        if self.neural_network:
            policy_probs, value = self.neural_network.predict(board)
            
            # Filter policy for legal actions
            total_prior = 0.0
            for action in legal_actions:
                action_idx = self.action_to_index(action)
                prior = policy_probs[action_idx] if action_idx < len(policy_probs) else 1.0 / len(legal_actions)
                node.children[action] = MCTSNode(prior=prior)
                total_prior += prior
            
            # Normalize priors
            if total_prior > 0:
                for child in node.children.values():
                    child.prior /= total_prior
        else:
            # Uniform priors if no network
            uniform_prior = 1.0 / len(legal_actions)
            for action in legal_actions:
                node.children[action] = MCTSNode(prior=uniform_prior)
    
    def backup(self, path: List[Tuple[MCTSNode, Any]], value: float):
        """
        Backup value through the search path
        
        Args:
            path: List of (node, action) pairs from selection
            value: Value to backup
        """
        for node, action in reversed(path):
            with self.lock:
                node.visit_count += 1
                node.value_sum += value
                node.virtual_loss -= 1
            
            # Flip value for opponent
            value = -value
    
    def add_dirichlet_noise(self, root: MCTSNode):
        """
        Add Dirichlet noise to root node for exploration
        
        Args:
            root: Root node to add noise to
        """
        if not root.children:
            return
        
        actions = list(root.children.keys())
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(actions))
        
        for action, noise_value in zip(actions, noise):
            child = root.children[action]
            child.prior = (1 - self.dirichlet_epsilon) * child.prior + self.dirichlet_epsilon * noise_value
    
    def select_action(self, root: MCTSNode) -> Tuple[Any, float]:
        """
        Select action based on visit counts and temperature
        
        Args:
            root: Root node to select from
            
        Returns:
            Tuple of (selected_action, estimated_value)
        """
        if not root.children:
            return None, 0.0
        
        actions = list(root.children.keys())
        visit_counts = [root.children[action].visit_count for action in actions]
        
        if self.temperature == 0:
            # Greedy selection
            best_idx = np.argmax(visit_counts)
            best_action = actions[best_idx]
        else:
            # Temperature-based selection
            visit_counts = np.array(visit_counts, dtype=float)
            if self.temperature != 1.0:
                visit_counts = visit_counts ** (1.0 / self.temperature)
            
            # Avoid division by zero
            if np.sum(visit_counts) == 0:
                probabilities = np.ones(len(visit_counts)) / len(visit_counts)
            else:
                probabilities = visit_counts / np.sum(visit_counts)
            
            best_idx = np.random.choice(len(actions), p=probabilities)
            best_action = actions[best_idx]
        
        estimated_value = root.children[best_action].value
        return best_action, estimated_value
    
    def get_or_create_node(self, state_hash: int) -> MCTSNode:
        """
        Get existing node from transposition table or create new one
        
        Args:
            state_hash: Hash of the board state
            
        Returns:
            MCTS node for this state
        """
        if self.use_transposition_table and state_hash in self.transposition_table:
            return self.transposition_table[state_hash]
        
        node = MCTSNode(state_hash=state_hash)
        if self.use_transposition_table:
            self.transposition_table[state_hash] = node
        
        return node
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """Get search statistics"""
        return {
            'nodes_searched': self.nodes_searched,
            'search_time': self.search_time,
            'nodes_per_second': self.nodes_searched / max(self.search_time, 0.001),
            'transposition_table_size': len(self.transposition_table)
        }
    
    def clear_search_statistics(self):
        """Clear search statistics"""
        self.nodes_searched = 0
        self.search_time = 0.0
    
    # Abstract methods to be implemented by chess-specific subclass
    def get_legal_actions(self, board) -> List[Any]:
        """Get list of legal actions from board state"""
        raise NotImplementedError
    
    def apply_action(self, board, action) -> Any:
        """Apply action to board and return new state"""
        raise NotImplementedError
    
    def is_terminal(self, board) -> bool:
        """Check if position is terminal (game over)"""
        raise NotImplementedError
    
    def get_game_result(self, board) -> float:
        """Get game result for terminal position"""
        raise NotImplementedError
    
    def evaluate_position(self, board) -> float:
        """Evaluate position using neural network or heuristic"""
        if self.neural_network:
            _, value = self.neural_network.predict(board)
            return value
        else:
            # Fallback heuristic evaluation
            return 0.0
    
    def action_to_index(self, action) -> int:
        """Convert action to network output index"""
        # Default implementation - should be overridden
        return hash(str(action)) % 4096


class ChessMCTS(AdvancedMCTS):
    """
    Chess-specific MCTS implementation
    """
    
    def __init__(self, chess_board, move_generator, **kwargs):
        super().__init__(**kwargs)
        self.chess_board = chess_board
        self.move_generator = move_generator
    
    def get_legal_actions(self, board) -> List[Any]:
        """Get list of legal chess moves"""
        try:
            if hasattr(self.move_generator, 'generate_legal_moves'):
                return self.move_generator.generate_legal_moves(board)
            else:
                # Fallback to basic move generation
                return []
        except Exception:
            return []
    
    def apply_action(self, board, action):
        """Apply chess move to board"""
        try:
            new_board = board.copy() if hasattr(board, 'copy') else board
            if hasattr(new_board, 'make_move'):
                new_board.make_move(action)
            return new_board
        except Exception:
            return board
    
    def is_terminal(self, board) -> bool:
        """Check if chess position is terminal"""
        try:
            if hasattr(board, 'is_game_over'):
                return board.is_game_over()
            else:
                # Check for basic terminal conditions
                legal_moves = self.get_legal_actions(board)
                return len(legal_moves) == 0
        except Exception:
            return False
    
    def get_game_result(self, board) -> float:
        """Get chess game result"""
        try:
            if hasattr(board, 'is_checkmate'):
                if board.is_checkmate():
                    return -1.0 if board.current_player == 'white' else 1.0
            
            if hasattr(board, 'is_stalemate'):
                if board.is_stalemate():
                    return 0.0
            
            # Default to draw
            return 0.0
        except Exception:
            return 0.0
    
    def action_to_index(self, action) -> int:
        """Convert chess move to network index"""
        try:
            # Simple encoding: from_square * 64 + to_square
            if hasattr(action, 'from_square') and hasattr(action, 'to_square'):
                return action.from_square * 64 + action.to_square
            else:
                return hash(str(action)) % 4096
        except Exception:
            return 0
