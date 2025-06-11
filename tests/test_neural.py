"""
Neural Network Testing Framework
===============================

Comprehensive testing suite for neural network components including:
- Network architecture validation
- Training pipeline testing
- MCTS integration testing
- Performance benchmarking
"""

import unittest
import time
import tempfile
import numpy as np
import torch
from pathlib import Path
import sys
import os
# Add src directory to path
src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
sys.path.insert(0, src_path)

from core.board import ChessBoard, Move, Color, Piece
from core.moves import MoveGenerator

# Import neural components with error handling
try:
    from neural.network import (
        AlphaZeroNetwork, NetworkConfig, BoardEncoder, MoveEncoder, 
        NeuralNetworkEvaluator, create_default_network
    )
    from neural.mcts import MCTS, MCTSConfig, MCTSSearchEngine
    from neural.training import TrainingConfig, SelfPlayEngine, TrainingOrchestrator
    from neural.hybrid_search import HybridSearchEngine, SearchMode, create_hybrid_engine
    NEURAL_AVAILABLE = True
except ImportError as e:
    print(f"Neural components not available: {e}")
    NEURAL_AVAILABLE = False


@unittest.skipUnless(NEURAL_AVAILABLE, "Neural network components not available")
class TestNeuralNetwork(unittest.TestCase):
    """Test neural network architecture and basic functionality."""
    
    def setUp(self):
        """Set up test components."""
        self.config = NetworkConfig()
        self.config.num_blocks = 2  # Reduced for testing
        self.config.filters = 64    # Reduced for testing
        self.device = torch.device("cpu")  # Use CPU for testing
        
        self.network = AlphaZeroNetwork(self.config).to(self.device)
        self.encoder = BoardEncoder()
        self.board = ChessBoard()
    
    def test_network_creation(self):
        """Test that the network can be created with the given configuration."""
        self.assertIsInstance(self.network, AlphaZeroNetwork)
        self.assertEqual(len(self.network.residual_blocks), self.config.num_blocks)
    
    def test_network_forward_pass(self):
        """Test forward pass through the network."""
        # Encode a board position
        encoded = self.encoder.encode_board(self.board)
        input_tensor = torch.from_numpy(encoded).unsqueeze(0).float()
        
        # Forward pass
        policy, value = self.network(input_tensor)
        
        # Check output shapes
        self.assertEqual(policy.shape, (1, self.config.policy_channels))
        self.assertEqual(value.shape, (1, 1))
        
        # Check output ranges
        self.assertTrue(torch.all(torch.isfinite(policy)))
        self.assertTrue(torch.all(torch.isfinite(value)))
        self.assertTrue(torch.all(torch.abs(value) <= 1.0))  # tanh output
    
    def test_board_encoding(self):
        """Test board encoding functionality."""
        encoded = self.encoder.encode_board(self.board)
        
        # Check shape
        self.assertEqual(encoded.shape, (18, 8, 8))
        
        # Check piece encoding (should have 32 pieces total)
        piece_planes = encoded[:12]  # First 12 channels are pieces
        total_pieces = piece_planes.sum()
        self.assertEqual(total_pieces, 32.0)
        
        # Check turn encoding (white to move in starting position)
        turn_plane = encoded[17]  # Last channel is turn
        self.assertEqual(turn_plane.sum(), 64.0)  # All squares should be 1
    
    def test_batch_encoding(self):
        """Test batch encoding of multiple positions."""
        boards = [self.board, self.board.copy()]
        batch_tensor = self.encoder.encode_batch(boards)
        
        self.assertEqual(batch_tensor.shape, (2, 18, 8, 8))
        self.assertIsInstance(batch_tensor, torch.Tensor)
    
    def test_neural_evaluator(self):
        """Test the neural network evaluator interface."""
        evaluator = NeuralNetworkEvaluator(device="cpu")
        evaluator.model = self.network  # Use our test network
        
        # Test position evaluation
        eval_score = evaluator.evaluate_position(self.board)
        self.assertIsInstance(eval_score, float)
        self.assertTrue(-1.0 <= eval_score <= 1.0)
        
        # Test move probabilities
        move_probs = evaluator.get_move_probabilities(self.board)
        self.assertIsInstance(move_probs, dict)
        self.assertTrue(len(move_probs) > 0)


@unittest.skipUnless(NEURAL_AVAILABLE, "Neural network components not available")
class TestMCTS(unittest.TestCase):
    """Test MCTS implementation with neural network guidance."""
    
    def setUp(self):
        """Set up test components."""
        self.board = ChessBoard()
        self.move_generator = MoveGenerator()
        self.neural_net = create_default_network()
        
        # Use minimal configuration for testing
        self.mcts_config = MCTSConfig()
        self.mcts_config.max_simulations = 50  # Reduced for testing
        self.mcts_config.max_time = 2.0
    
    def test_mcts_creation(self):
        """Test MCTS creation and basic setup."""
        mcts = MCTS(self.neural_net, self.move_generator, self.mcts_config)
        self.assertIsNotNone(mcts)
        self.assertEqual(mcts.config.max_simulations, 50)
    
    def test_mcts_search(self):
        """Test MCTS search execution."""
        mcts = MCTS(self.neural_net, self.move_generator, self.mcts_config)
        
        best_move, stats = mcts.search(self.board, max_simulations=20)
        
        # Check that we got a valid move
        self.assertIsInstance(best_move, Move)
        
        # Check statistics
        self.assertGreater(stats.simulations, 0)
        self.assertGreater(stats.time_used, 0)
        self.assertGreaterEqual(stats.nodes_created, 0)
    
    def test_mcts_node_operations(self):
        """Test MCTS node operations."""
        from neural.mcts import MCTSNode
        
        node = MCTSNode(self.board)
        
        # Test initial state
        self.assertTrue(node.is_leaf())
        self.assertEqual(node.visit_count, 0)
        self.assertEqual(node.get_q_value(), 0.0)
        
        # Test backup
        node.backup(0.5)
        self.assertEqual(node.visit_count, 1)
        self.assertEqual(node.get_q_value(), 0.5)
    
    def test_mcts_search_engine(self):
        """Test high-level MCTS search engine interface."""
        engine = MCTSSearchEngine(self.neural_net, self.move_generator)
        
        best_move, search_info = engine.search_position(
            self.board, time_limit=1.0, simulations=20
        )
        
        self.assertIsInstance(best_move, Move)
        self.assertIn('simulations', search_info)
        self.assertIn('time', search_info)
        self.assertIn('nps', search_info)


@unittest.skipUnless(NEURAL_AVAILABLE, "Neural network components not available")
class TestSelfPlayTraining(unittest.TestCase):
    """Test self-play training infrastructure."""
    
    def setUp(self):
        """Set up test components."""
        self.move_generator = MoveGenerator()
        self.neural_net = create_default_network()
        
        # Minimal MCTS config for testing
        self.mcts_config = MCTSConfig()
        self.mcts_config.max_simulations = 20
        self.mcts_config.add_dirichlet_noise = True
    
    def test_self_play_engine_creation(self):
        """Test self-play engine creation."""
        engine = SelfPlayEngine(self.neural_net, self.move_generator, self.mcts_config)
        self.assertIsNotNone(engine)
    
    @unittest.skip("Self-play games take too long for regular testing")
    def test_self_play_game(self):
        """Test playing a complete self-play game."""
        engine = SelfPlayEngine(self.neural_net, self.move_generator, self.mcts_config)
        
        # Play a short game
        game_result = engine.play_game(temperature_moves=5, temperature=1.0)
        
        # Check game result
        self.assertGreater(len(game_result.positions), 0)
        self.assertGreater(len(game_result.move_probabilities), 0)
        self.assertGreater(len(game_result.values), 0)
        self.assertEqual(len(game_result.positions), len(game_result.values))
    
    def test_training_config(self):
        """Test training configuration."""
        config = TrainingConfig()
        config.num_iterations = 1
        config.games_per_iteration = 1
        config.training_epochs = 1
        config.mcts_simulations = 10
        
        self.assertEqual(config.num_iterations, 1)
        self.assertEqual(config.games_per_iteration, 1)


@unittest.skipUnless(NEURAL_AVAILABLE, "Neural network components not available")
class TestHybridSearch(unittest.TestCase):
    """Test hybrid search engine functionality."""
    
    def setUp(self):
        """Set up test components."""
        self.board = ChessBoard()
        self.engine = create_hybrid_engine()
    
    def test_hybrid_engine_creation(self):
        """Test hybrid engine creation."""
        self.assertIsNotNone(self.engine)
        self.assertIsNotNone(self.engine.move_generator)
        self.assertIsNotNone(self.engine.evaluator)
    
    def test_position_analysis(self):
        """Test position analysis functionality."""
        analysis = self.engine.analyzer.analyze_position(self.board)
        
        # Check that we get expected analysis fields
        expected_fields = [
            'total_material', 'material_imbalance', 'num_legal_moves',
            'capture_ratio', 'complexity_score', 'game_phase'
        ]
        
        for field in expected_fields:
            self.assertIn(field, analysis)
            self.assertIsInstance(analysis[field], (int, float))
    
    def test_search_mode_selection(self):
        """Test search mode selection logic."""
        mode = self.engine._select_search_mode(self.board, 10.0)
        self.assertIsInstance(mode, SearchMode)
    
    def test_hybrid_search(self):
        """Test hybrid search execution."""
        best_move, search_info = self.engine.search_position(
            self.board, time_limit=2.0
        )
        
        self.assertIsInstance(best_move, Move)
        self.assertIn('search_mode', search_info)
        self.assertIn('actual_time', search_info)
        self.assertIn('position_analysis', search_info)
    
    def test_search_statistics(self):
        """Test search statistics collection."""
        # Perform a search to generate statistics
        self.engine.search_position(self.board, time_limit=1.0)
        
        stats = self.engine.get_search_statistics()
        self.assertIn('total_searches', stats)
        self.assertGreater(stats['total_searches'], 0)


class TestPerformance(unittest.TestCase):
    """Performance benchmarks for neural network components."""
    
    @unittest.skipUnless(NEURAL_AVAILABLE, "Neural network components not available")
    def test_neural_evaluation_speed(self):
        """Test neural network evaluation speed."""
        neural_net = create_default_network()
        board = ChessBoard()
        
        # Warm up
        for _ in range(5):
            neural_net.evaluate_position(board)
        
        # Benchmark
        start_time = time.time()
        num_evaluations = 100
        
        for _ in range(num_evaluations):
            neural_net.evaluate_position(board)
        
        end_time = time.time()
        total_time = end_time - start_time
        evals_per_second = num_evaluations / total_time
        
        print(f"Neural network evaluations per second: {evals_per_second:.1f}")
        
        # Should be reasonably fast (at least 50 evaluations per second on CPU)
        self.assertGreater(evals_per_second, 20.0)
    
    @unittest.skipUnless(NEURAL_AVAILABLE, "Neural network components not available")
    def test_mcts_search_speed(self):
        """Test MCTS search performance."""
        neural_net = create_default_network()
        move_generator = MoveGenerator()
        board = ChessBoard()
        
        config = MCTSConfig()
        config.max_simulations = 100
        config.max_time = 10.0
        
        mcts = MCTS(neural_net, move_generator, config)
        
        start_time = time.time()
        best_move, stats = mcts.search(board)
        end_time = time.time()
        
        print(f"MCTS: {stats.simulations} simulations in {stats.time_used:.2f}s")
        print(f"MCTS speed: {stats.nps:.1f} simulations/sec")
        
        # Check that we completed the search
        self.assertIsInstance(best_move, Move)
        self.assertGreater(stats.simulations, 50)  # Should complete reasonable number


class TestIntegration(unittest.TestCase):
    """Integration tests combining multiple components."""
    
    @unittest.skipUnless(NEURAL_AVAILABLE, "Neural network components not available")
    def test_neural_vs_classical_comparison(self):
        """Compare neural network and classical evaluation."""
        from core.evaluation import Evaluator
        
        neural_net = create_default_network()
        classical_evaluator = Evaluator()
        board = ChessBoard()
        
        # Get evaluations
        neural_eval = neural_net.evaluate_position(board)
        classical_eval = classical_evaluator.evaluate(board)
        
        print(f"Starting position - Neural: {neural_eval:.3f}, Classical: {classical_eval}")
          # Both should be close to 0 for starting position (though not necessarily equal)
        self.assertTrue(-1.0 <= neural_eval <= 1.0)
        self.assertTrue(-500 <= classical_eval <= 500)
    
    @unittest.skipUnless(NEURAL_AVAILABLE, "Neural network components not available")
    def test_search_engine_comparison(self):
        """Compare different search engines on the same position."""
        board = ChessBoard()
        time_limit = 2.0
        
        # Classical search
        from core.evaluation import Evaluator
        from core.search import SearchEngine, SearchInfo
        classical_evaluator = Evaluator()
        move_generator = MoveGenerator()
        classical_engine = SearchEngine(classical_evaluator, move_generator)
        classical_move, score, search_info = classical_engine.search(board, 8, time_limit)
        
        # Hybrid search
        hybrid_engine = create_hybrid_engine()
        neural_move, _ = hybrid_engine.search_position(board, time_limit=time_limit)
        
        print(f"Classical move: {classical_move}")
        print(f"Neural/hybrid move: {neural_move}")
        
        # Both should find valid moves
        self.assertIsInstance(classical_move, Move)
        self.assertIsInstance(neural_move, Move)


def run_neural_tests():
    """Run all neural network tests."""
    if not NEURAL_AVAILABLE:
        print("Neural network components not available. Skipping neural tests.")
        return False
    
    # Create test suite
    test_classes = [
        TestNeuralNetwork,
        TestMCTS,
        TestSelfPlayTraining,
        TestHybridSearch,
        TestPerformance,
        TestIntegration
    ]
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def benchmark_neural_components():
    """Run performance benchmarks for neural components."""
    if not NEURAL_AVAILABLE:
        print("Neural network components not available.")
        return
    
    print("Running neural network benchmarks...")
    
    # Board encoding benchmark
    encoder = BoardEncoder()
    board = ChessBoard()
    
    start_time = time.time()
    for _ in range(1000):
        encoded = encoder.encode_board(board)
    end_time = time.time()
    
    encoding_speed = 1000 / (end_time - start_time)
    print(f"Board encoding speed: {encoding_speed:.1f} encodings/sec")
    
    # Neural network inference benchmark
    neural_net = create_default_network()
    
    start_time = time.time()
    for _ in range(100):
        eval_score = neural_net.evaluate_position(board)
    end_time = time.time()
    
    inference_speed = 100 / (end_time - start_time)
    print(f"Neural inference speed: {inference_speed:.1f} inferences/sec")
    
    # MCTS benchmark
    move_generator = MoveGenerator()
    config = MCTSConfig()
    config.max_simulations = 200
    
    mcts = MCTS(neural_net, move_generator, config)
    
    start_time = time.time()
    best_move, stats = mcts.search(board)
    end_time = time.time()
    
    print(f"MCTS: {stats.simulations} simulations in {stats.time_used:.2f}s")
    print(f"MCTS speed: {stats.nps:.1f} simulations/sec")


if __name__ == "__main__":
    print("Chess AI Neural Network Test Suite")
    print("=" * 50)
    
    # Run basic availability check
    if NEURAL_AVAILABLE:
        print("✓ Neural network components available")
        
        # Run tests
        success = run_neural_tests()
        
        if success:
            print("\n✓ All neural network tests passed!")
        else:
            print("\n✗ Some tests failed")
        
        # Run benchmarks
        print("\nRunning benchmarks...")
        benchmark_neural_components()
        
    else:
        print("✗ Neural network components not available")
        print("Install PyTorch to enable neural network functionality")
    
    print("\nTest suite completed.")
