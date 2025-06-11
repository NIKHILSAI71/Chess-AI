"""
Hybrid Search Engine
===================

Advanced chess search engine combining traditional alpha-beta search with neural network evaluation.
Implements multiple search strategies for optimal performance across different game phases.
"""

import time
import threading
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.board import ChessBoard, Move, Color
from core.moves import MoveGenerator
from core.evaluation import Evaluator
from core.search import SearchEngine, SearchInfo
from neural.network import NeuralNetworkEvaluator
from neural.mcts import MCTSSearchEngine, MCTSConfig


class SearchMode(Enum):
    """Search engine modes."""
    CLASSICAL = "classical"           # Traditional alpha-beta search
    NEURAL_MCTS = "neural_mcts"      # MCTS with neural network
    HYBRID = "hybrid"                 # Adaptive hybrid approach
    NEURAL_AB = "neural_ab"          # Alpha-beta with NN evaluation


@dataclass
class HybridSearchConfig:
    """Configuration for hybrid search engine."""
    # Search mode selection
    default_mode: SearchMode = SearchMode.HYBRID
    
    # Classical search parameters
    max_depth: int = 12
    use_quiescence: bool = True
    use_transposition_table: bool = True
    
    # MCTS parameters
    mcts_simulations: int = 800
    mcts_time_limit: float = 10.0
    
    # Hybrid decision parameters
    complexity_threshold: float = 0.5    # When to switch to MCTS
    time_threshold: float = 5.0          # Time threshold for mode selection
    material_threshold: int = 1500       # Material threshold for endgame
    
    # Neural network parameters
    use_neural_eval: bool = True
    nn_eval_threshold: int = 8           # Depth to start using NN eval
    
    # Performance parameters
    move_time: float = 10.0              # Time per move
    increment: float = 0.1               # Time increment
    moves_to_go: Optional[int] = None    # Moves until time control


class PositionAnalyzer:
    """
    Analyzes chess positions to determine optimal search strategy.
    
    Evaluates position characteristics like complexity, material, and game phase
    to decide between different search approaches.
    """
    
    def __init__(self, move_generator: MoveGenerator, evaluator: Evaluator):
        self.move_generator = move_generator
        self.evaluator = evaluator
    
    def analyze_position(self, board: ChessBoard) -> Dict[str, float]:
        """
        Analyze a chess position and return characteristics.
        
        Args:
            board: Position to analyze
            
        Returns:
            Dictionary of position characteristics
        """
        legal_moves = self.move_generator.generate_legal_moves(board)
        captures = self.move_generator.generate_captures(board)
        
        # Material analysis
        material_white = self._count_material(board, Color.WHITE)
        material_black = self._count_material(board, Color.BLACK)
        total_material = material_white + material_black
        material_imbalance = abs(material_white - material_black)
        
        # Tactical complexity
        num_legal_moves = len(legal_moves)
        num_captures = len(captures)
        capture_ratio = num_captures / max(num_legal_moves, 1)
        
        # King safety
        white_king_safety = self._evaluate_king_safety(board, Color.WHITE)
        black_king_safety = self._evaluate_king_safety(board, Color.BLACK)
        
        # Position evaluation
        position_eval = self.evaluator.evaluate(board)
        eval_magnitude = abs(position_eval)
        
        # Game phase
        game_phase = self._determine_game_phase(total_material)
        
        return {
            'total_material': total_material,
            'material_imbalance': material_imbalance,
            'num_legal_moves': num_legal_moves,
            'num_captures': num_captures,
            'capture_ratio': capture_ratio,
            'king_safety_white': white_king_safety,
            'king_safety_black': black_king_safety,
            'position_eval': position_eval,
            'eval_magnitude': eval_magnitude,
            'game_phase': game_phase,
            'complexity_score': self._calculate_complexity(
                num_legal_moves, capture_ratio, eval_magnitude, game_phase
            )
        }
    
    def _count_material(self, board: ChessBoard, color: Color) -> int:
        """Count material for a given color."""
        from core.board import Piece
        
        material = 0
        piece_values = {
            Piece.PAWN: 100,
            Piece.KNIGHT: 300,
            Piece.BISHOP: 300,
            Piece.ROOK: 500,
            Piece.QUEEN: 900,
            Piece.KING: 0
        }
        
        for piece, value in piece_values.items():
            count = bin(board.bitboards[(color, piece)].value).count('1')
            material += count * value
        
        return material
    
    def _evaluate_king_safety(self, board: ChessBoard, color: Color) -> float:
        """Evaluate king safety (simplified)."""
        king_square = board.get_king_square(color)
        if king_square == -1:
            return 0.0
        
        # Simple king safety based on position and surrounding pieces
        file = king_square % 8
        rank = king_square // 8
        
        safety = 0.0
        
        # Prefer castled position
        if color == Color.WHITE:
            if file >= 6 or file <= 1:  # Castled kingside or queenside
                safety += 0.5
        else:
            if file >= 6 or file <= 1:
                safety += 0.5
        
        # Penalize exposed king
        if color == Color.WHITE and rank > 2:
            safety -= 0.3
        elif color == Color.BLACK and rank < 5:
            safety -= 0.3
        
        return safety
    
    def _determine_game_phase(self, total_material: int) -> float:
        """
        Determine game phase based on material.
        
        Returns:
            0.0 = endgame, 1.0 = opening/middlegame
        """
        max_material = 7800  # Approximate material at start
        return min(1.0, total_material / max_material)
    
    def _calculate_complexity(self, num_moves: int, capture_ratio: float, 
                            eval_magnitude: float, game_phase: float) -> float:
        """Calculate overall position complexity."""
        # More moves = more complex
        move_complexity = min(1.0, num_moves / 50.0)
        
        # More captures = more tactical
        tactical_complexity = capture_ratio
        
        # Sharp evaluation = complex position
        eval_complexity = min(1.0, eval_magnitude / 500.0)
        
        # Middlegame is typically more complex
        phase_complexity = game_phase
        
        # Weighted combination
        complexity = (0.3 * move_complexity + 
                     0.3 * tactical_complexity + 
                     0.2 * eval_complexity + 
                     0.2 * phase_complexity)
        
        return complexity


class HybridSearchEngine:
    """
    Hybrid search engine that adaptively chooses between different search strategies.
    
    Combines classical alpha-beta search, MCTS with neural networks, and hybrid approaches
    based on position characteristics and time constraints.
    """
    
    def __init__(self, move_generator: MoveGenerator, evaluator: Evaluator,
                 neural_net: Optional[NeuralNetworkEvaluator] = None,
                 config: HybridSearchConfig = None):
        
        self.move_generator = move_generator
        self.evaluator = evaluator
        self.neural_net = neural_net
        self.config = config or HybridSearchConfig()
        
        # Initialize search engines
        self.classical_engine = SearchEngine(evaluator, move_generator)
        
        if neural_net:
            self.mcts_engine = MCTSSearchEngine(neural_net, move_generator)
            self.neural_evaluator = neural_net
        else:
            self.mcts_engine = None
            self.neural_evaluator = None
        
        # Position analyzer
        self.analyzer = PositionAnalyzer(move_generator, evaluator)
        
        # Search state
        self.current_mode = self.config.default_mode
        self.search_history = []
        self.performance_stats = {}
    
    def search_position(self, board: ChessBoard, time_limit: float = None,
                       depth: int = None, mode: SearchMode = None) -> Tuple[Move, Dict]:
        """
        Search for the best move in the given position.
        
        Args:
            board: Current board position
            time_limit: Maximum search time
            depth: Search depth (for classical search)
            mode: Force specific search mode
            
        Returns:
            Tuple of (best_move, search_info)
        """
        start_time = time.time()
        
        # Determine search parameters
        time_limit = time_limit or self.config.move_time
        depth = depth or self.config.max_depth
        mode = mode or self._select_search_mode(board, time_limit)
        
        # Analyze position
        position_analysis = self.analyzer.analyze_position(board)
        
        # Execute search based on selected mode
        if mode == SearchMode.CLASSICAL:
            best_move, search_info = self._search_classical(board, depth, time_limit)
        elif mode == SearchMode.NEURAL_MCTS and self.mcts_engine:
            best_move, search_info = self._search_mcts(board, time_limit)
        elif mode == SearchMode.NEURAL_AB and self.neural_evaluator:
            best_move, search_info = self._search_neural_ab(board, depth, time_limit)
        elif mode == SearchMode.HYBRID:
            best_move, search_info = self._search_hybrid(board, depth, time_limit)
        else:
            # Fallback to classical search
            best_move, search_info = self._search_classical(board, depth, time_limit)
        
        # Update search info
        search_time = time.time() - start_time
        search_info.update({
            'search_mode': mode.value,
            'actual_time': search_time,
            'position_analysis': position_analysis
        })
        
        # Record search history
        self._record_search(mode, search_time, position_analysis, search_info)
        
        return best_move, search_info
    
    def _select_search_mode(self, board: ChessBoard, time_limit: float) -> SearchMode:
        """
        Intelligently select the best search mode for the position.
        
        Args:
            board: Current position
            time_limit: Available time
            
        Returns:
            Selected search mode
        """
        # Analyze position characteristics
        analysis = self.analyzer.analyze_position(board)
        
        # Decision factors
        complexity = analysis['complexity_score']
        material = analysis['total_material']
        captures = analysis['capture_ratio']
        game_phase = analysis['game_phase']
        
        # Mode selection logic
        if not self.neural_net:
            return SearchMode.CLASSICAL
        
        # Endgame: prefer classical search for precise calculation
        if material < self.config.material_threshold:
            return SearchMode.CLASSICAL
        
        # High tactical complexity: use MCTS for better exploration
        if complexity > self.config.complexity_threshold and captures > 0.3:
            if time_limit >= self.config.time_threshold:
                return SearchMode.NEURAL_MCTS
        
        # Opening/middlegame with sufficient time: hybrid approach
        if game_phase > 0.5 and time_limit >= self.config.time_threshold:
            return SearchMode.HYBRID
          # Limited time: classical search (faster)
        if time_limit < self.config.time_threshold:
            return SearchMode.NEURAL_AB if self.neural_evaluator else SearchMode.CLASSICAL
        
        # Default to hybrid
        return SearchMode.HYBRID
    
    def _search_classical(self, board: ChessBoard, depth: int, 
                         time_limit: float) -> Tuple[Move, Dict]:
        """Execute classical alpha-beta search."""
        best_move, score, search_info = self.classical_engine.search(board, depth, time_limit)
        
        return best_move, {
            'engine': 'classical',
            'depth': search_info.depth,
            'nodes': search_info.nodes,
            'time': time.time() - search_info.time_start if search_info.time_start > 0 else 0,
            'evaluation': score,
            'pv': search_info.pv
        }
    
    def _search_mcts(self, board: ChessBoard, time_limit: float) -> Tuple[Move, Dict]:
        """Execute MCTS search with neural network."""
        best_move, search_info = self.mcts_engine.search_position(
            board, time_limit=time_limit, simulations=self.config.mcts_simulations
        )
        
        return best_move, {
            'engine': 'mcts',
            'simulations': search_info['simulations'],
            'time': search_info['time'],
            'nps': search_info['nps'],
            'pv': search_info['principal_variation']
        }
    
    def _search_neural_ab(self, board: ChessBoard, depth: int,
                         time_limit: float) -> Tuple[Move, Dict]:
        """Execute alpha-beta search with neural network evaluation."""
        # Temporarily replace evaluator
        original_evaluator = self.classical_engine.evaluator
        
        # Create neural network wrapper for classical search
        class NeuralEvaluatorWrapper:
            def __init__(self, neural_net):
                self.neural_net = neural_net
            
            def evaluate(self, board):
                return int(self.neural_net.evaluate_position(board) * 100)
        
        # Use neural evaluation for deeper searches
        if depth >= self.config.nn_eval_threshold:
            self.classical_engine.evaluator = NeuralEvaluatorWrapper(self.neural_evaluator)
        
        try:
            search_info = SearchInfo()
            search_info.max_depth = depth
            search_info.time_limit = time_limit
            
            best_move = self.classical_engine.search(board, search_info)
            
            return best_move, {
                'engine': 'neural_ab',
                'depth': search_info.depth_reached,
                'nodes': search_info.nodes_searched,
                'time': search_info.time_used,
                'evaluation': search_info.best_score,
                'pv': search_info.principal_variation
            }
        finally:
            # Restore original evaluator
            self.classical_engine.evaluator = original_evaluator
    
    def _search_hybrid(self, board: ChessBoard, depth: int,
                      time_limit: float) -> Tuple[Move, Dict]:
        """
        Execute hybrid search combining multiple approaches.
        
        Uses time allocation between classical and MCTS search.
        """
        # Allocate time between different approaches
        classical_time = time_limit * 0.3
        mcts_time = time_limit * 0.7
        
        results = []
        
        # Quick classical search for baseline
        classical_move, classical_info = self._search_classical(
            board, min(depth, 8), classical_time
        )
        results.append(('classical', classical_move, classical_info))
        
        # MCTS search for position understanding
        if self.mcts_engine and mcts_time > 1.0:
            mcts_move, mcts_info = self._search_mcts(board, mcts_time)
            results.append(('mcts', mcts_move, mcts_info))
        
        # Select best result based on multiple criteria
        best_move, best_info = self._select_best_hybrid_result(results, board)
        
        return best_move, {
            'engine': 'hybrid',
            'results': results,
            'selected': best_info,
            'time': time_limit
        }
    
    def _select_best_hybrid_result(self, results: List[Tuple[str, Move, Dict]],
                                  board: ChessBoard) -> Tuple[Move, Dict]:
        """Select the best result from hybrid search."""
        if not results:
            return None, {}
        
        # Simple selection: prefer MCTS if available, otherwise classical
        for engine_type, move, info in results:
            if engine_type == 'mcts' and move:
                return move, info
        
        # Fallback to first available result
        return results[0][1], results[0][2]
    
    def _record_search(self, mode: SearchMode, time_used: float,
                      position_analysis: Dict, search_info: Dict):
        """Record search statistics for performance analysis."""
        record = {
            'mode': mode.value,
            'time': time_used,
            'complexity': position_analysis.get('complexity_score', 0),
            'material': position_analysis.get('total_material', 0),
            'nodes': search_info.get('nodes', 0),
            'depth': search_info.get('depth', 0)
        }
        
        self.search_history.append(record)
        
        # Keep only recent history
        if len(self.search_history) > 100:
            self.search_history = self.search_history[-100:]
    
    def get_search_statistics(self) -> Dict:
        """Get detailed search performance statistics."""
        if not self.search_history:
            return {}
        
        # Analyze performance by mode
        mode_stats = {}
        for record in self.search_history:
            mode = record['mode']
            if mode not in mode_stats:
                mode_stats[mode] = {
                    'count': 0,
                    'total_time': 0,
                    'total_nodes': 0,
                    'avg_depth': 0
                }
            
            stats = mode_stats[mode]
            stats['count'] += 1
            stats['total_time'] += record['time']
            stats['total_nodes'] += record.get('nodes', 0)
            stats['avg_depth'] += record.get('depth', 0)
        
        # Calculate averages
        for mode, stats in mode_stats.items():
            if stats['count'] > 0:
                stats['avg_time'] = stats['total_time'] / stats['count']
                stats['avg_nodes'] = stats['total_nodes'] / stats['count']
                stats['avg_depth'] = stats['avg_depth'] / stats['count']
                stats['nps'] = stats['avg_nodes'] / max(stats['avg_time'], 0.001)
        
        return {
            'total_searches': len(self.search_history),
            'mode_statistics': mode_stats,
            'current_mode': self.current_mode.value
        }
    
    def configure(self, **kwargs):
        """Update search configuration."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
    
    def set_time_control(self, time_remaining: float, increment: float = 0,
                        moves_to_go: Optional[int] = None):
        """Set time control parameters."""
        self.config.move_time = self._calculate_move_time(
            time_remaining, increment, moves_to_go
        )
    
    def _calculate_move_time(self, time_remaining: float, increment: float,
                           moves_to_go: Optional[int]) -> float:
        """Calculate optimal time allocation for the current move."""
        if moves_to_go:
            # Time control with specific number of moves
            base_time = time_remaining / (moves_to_go + 2)
        else:
            # Increment time control
            base_time = time_remaining / 30  # Assume ~30 moves remaining
        
        # Add increment
        move_time = base_time + increment * 0.8
        
        # Ensure minimum and maximum time
        move_time = max(0.1, min(move_time, time_remaining * 0.1))
        
        return move_time


# Factory function for creating search engines
def create_hybrid_engine(neural_net_path: Optional[str] = None) -> HybridSearchEngine:
    """
    Create a hybrid search engine with optional neural network.
    
    Args:
        neural_net_path: Path to trained neural network model
        
    Returns:
        Configured hybrid search engine
    """
    from core.moves import MoveGenerator
    from core.evaluation import Evaluator
    from neural.network import NeuralNetworkEvaluator
    
    # Initialize core components
    move_generator = MoveGenerator()
    evaluator = Evaluator()
    
    # Initialize neural network if available
    neural_net = None
    if neural_net_path:
        try:
            neural_net = NeuralNetworkEvaluator(neural_net_path)
            print(f"Loaded neural network from {neural_net_path}")
        except Exception as e:
            print(f"Failed to load neural network: {e}")
            print("Using classical evaluation only")
    
    # Create hybrid engine
    config = HybridSearchConfig()
    engine = HybridSearchEngine(move_generator, evaluator, neural_net, config)
    
    return engine


# Example usage and testing
if __name__ == "__main__":
    print("Testing Hybrid Search Engine...")
    
    # Create engine
    engine = create_hybrid_engine()
    
    # Test position analysis
    from core.board import ChessBoard
    board = ChessBoard()
    
    print("Analyzing starting position...")
    analysis = engine.analyzer.analyze_position(board)
    for key, value in analysis.items():
        print(f"  {key}: {value:.3f}")
    
    # Test search mode selection
    selected_mode = engine._select_search_mode(board, 10.0)
    print(f"Selected search mode: {selected_mode.value}")
    
    # Test quick search
    print("Performing quick search...")
    best_move, search_info = engine.search_position(board, time_limit=2.0)
    
    if best_move:
        print(f"Best move: {best_move}")
        print(f"Search engine: {search_info.get('engine')}")
        print(f"Time used: {search_info.get('actual_time', 0):.2f}s")
    
    # Test statistics
    stats = engine.get_search_statistics()
    print(f"Search statistics: {stats}")
    
    print("Hybrid search engine tests completed!")
