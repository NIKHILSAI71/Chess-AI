"""
Chess Engine Core Tests
======================

Comprehensive test suite for validating the chess engine's core functionality.
Tests include board representation, move generation, search algorithms, and UCI protocol.
"""

import unittest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.board import ChessBoard, Move, Piece, Color, Square, BitBoard
from core.evaluation import Evaluator
from core.moves import MoveGenerator, AttackTables
from core.search import SearchEngine, TranspositionTable
from utils.fen import FENParser, STARTING_FEN
from utils.zobrist import ZobristHasher


class TestBitBoard(unittest.TestCase):
    """Test BitBoard operations."""
    
    def test_basic_operations(self):
        """Test basic bitboard operations."""
        bb1 = BitBoard(0xFF)  # First rank
        bb2 = BitBoard(0xFF00)  # Second rank
        
        # Test OR operation
        combined = bb1 | bb2
        self.assertEqual(combined.value, 0xFFFF)
        
        # Test AND operation
        intersection = bb1 & bb2
        self.assertEqual(intersection.value, 0)
        
        # Test XOR operation
        xor_result = bb1 ^ bb2
        self.assertEqual(xor_result.value, 0xFFFF)
        
        # Test inversion
        inverted = ~bb1
        self.assertEqual(inverted.value, ~0xFF & 0xFFFFFFFFFFFFFFFF)
    
    def test_bit_manipulation(self):
        """Test bit manipulation methods."""
        bb = BitBoard(0)
        
        # Test set_bit
        bb = bb.set_bit(0)
        self.assertTrue(bb.get_bit(0))
        self.assertFalse(bb.get_bit(1))
        
        # Test clear_bit
        bb = bb.clear_bit(0)
        self.assertFalse(bb.get_bit(0))
        
        # Test population count
        bb = BitBoard(0xFF)  # 8 bits set
        self.assertEqual(bb.pop_count(), 8)
        
        # Test least significant bit
        bb = BitBoard(0x8)  # Bit 3 set
        self.assertEqual(bb.ls1b(), 3)


class TestChessBoard(unittest.TestCase):
    """Test ChessBoard representation and basic operations."""
    
    def setUp(self):
        """Set up test board."""
        self.board = ChessBoard()
    
    def test_initial_position(self):
        """Test initial board setup."""
        # Check initial piece placement
        self.assertEqual(self.board.get_piece_at(Square.E1), (Piece.KING, Color.WHITE))
        self.assertEqual(self.board.get_piece_at(Square.E8), (Piece.KING, Color.BLACK))
        self.assertEqual(self.board.get_piece_at(Square.A1), (Piece.ROOK, Color.WHITE))
        self.assertEqual(self.board.get_piece_at(Square.H8), (Piece.ROOK, Color.BLACK))
        
        # Check empty squares
        self.assertEqual(self.board.get_piece_at(Square.E4), (Piece.EMPTY, Color.WHITE))
        
        # Check game state
        self.assertEqual(self.board.turn, Color.WHITE)
        self.assertTrue(self.board.castling_rights.white_kingside)
        self.assertTrue(self.board.castling_rights.black_queenside)
        self.assertIsNone(self.board.en_passant_target)
    
    def test_fen_conversion(self):
        """Test FEN string conversion."""
        # Test starting position FEN
        fen = self.board.to_fen()
        self.assertEqual(fen, STARTING_FEN)
        
        # Test FEN parsing
        test_fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
        board_from_fen = ChessBoard.from_fen(test_fen)
        self.assertEqual(board_from_fen.to_fen(), test_fen)
    
    def test_piece_placement(self):
        """Test piece placement and removal."""
        # Clear a square and check
        self.board.clear_square(Square.E2)
        self.assertEqual(self.board.get_piece_at(Square.E2), (Piece.EMPTY, Color.WHITE))
        
        # Place a piece and check
        self.board.set_piece_at(Square.E4, Piece.PAWN, Color.WHITE)
        self.assertEqual(self.board.get_piece_at(Square.E4), (Piece.PAWN, Color.WHITE))
    
    def test_move_making(self):
        """Test basic move making."""
        # Test pawn move
        move = Move(Square.E2, Square.E4, Piece.PAWN)
        self.assertTrue(self.board.make_move(move))
        
        # Check piece moved
        self.assertEqual(self.board.get_piece_at(Square.E2), (Piece.EMPTY, Color.WHITE))
        self.assertEqual(self.board.get_piece_at(Square.E4), (Piece.PAWN, Color.WHITE))
        
        # Check turn changed
        self.assertEqual(self.board.turn, Color.BLACK)
        
        # Check en passant square set
        self.assertEqual(self.board.en_passant_target, Square.E3)
    
    def test_castling_moves(self):
        """Test castling move handling."""
        # Clear squares for castling
        self.board.clear_square(Square.F1)
        self.board.clear_square(Square.G1)
        
        # Make castling move
        castling_move = Move(Square.E1, Square.G1, Piece.KING, is_castling=True)
        self.board.make_move(castling_move)
        
        # Check king and rook positions
        self.assertEqual(self.board.get_piece_at(Square.G1), (Piece.KING, Color.WHITE))
        self.assertEqual(self.board.get_piece_at(Square.F1), (Piece.ROOK, Color.WHITE))
        self.assertEqual(self.board.get_piece_at(Square.E1), (Piece.EMPTY, Color.WHITE))
        self.assertEqual(self.board.get_piece_at(Square.H1), (Piece.EMPTY, Color.WHITE))


class TestMoveGeneration(unittest.TestCase):
    """Test move generation functionality."""
    
    def setUp(self):
        """Set up test components."""
        self.board = ChessBoard()
        self.move_generator = MoveGenerator()
        self.attack_tables = AttackTables()
    
    def test_attack_tables_initialization(self):
        """Test attack tables are properly initialized."""
        # Test knight attacks
        knight_attacks = self.attack_tables.knight_attacks[Square.E4]
        self.assertTrue(knight_attacks.get_bit(Square.F6))
        self.assertTrue(knight_attacks.get_bit(Square.D6))
        self.assertFalse(knight_attacks.get_bit(Square.E5))
        
        # Test king attacks
        king_attacks = self.attack_tables.king_attacks[Square.E4]
        self.assertTrue(king_attacks.get_bit(Square.E5))
        self.assertTrue(king_attacks.get_bit(Square.F5))
        self.assertFalse(king_attacks.get_bit(Square.E6))
    
    def test_pawn_move_generation(self):
        """Test pawn move generation."""
        # Test initial pawn moves (should include double push)
        moves = list(self.move_generator.generate_pawn_moves(self.board, Color.WHITE))
        
        # Should have 16 pawn moves (8 single + 8 double pushes)
        pawn_moves = [m for m in moves if m.piece == Piece.PAWN]
        self.assertEqual(len(pawn_moves), 16)
        
        # Check specific moves exist
        e2_e3 = Move(Square.E2, Square.E3, Piece.PAWN)
        e2_e4 = Move(Square.E2, Square.E4, Piece.PAWN)
        self.assertIn(e2_e3, pawn_moves)
        self.assertIn(e2_e4, pawn_moves)


class TestEvaluation(unittest.TestCase):
    """Test position evaluation."""
    
    def setUp(self):
        """Set up test components."""
        self.board = ChessBoard()
        self.evaluator = Evaluator()
    
    def test_initial_position_evaluation(self):
        """Test evaluation of starting position."""
        score = self.evaluator.evaluate(self.board)
        # Starting position should be roughly equal (small advantage to white for tempo)
        self.assertAlmostEqual(score, 0, delta=50)
    
    def test_material_evaluation(self):
        """Test material-based evaluation."""
        # Remove black queen
        self.board.clear_square(Square.D8)
        score = self.evaluator.evaluate(self.board)
        
        # White should have significant advantage
        self.assertGreater(score, 800)  # Queen value is 900
    
    def test_piece_square_tables(self):
        """Test piece-square table evaluation."""
        # Move pawn to center
        self.board.clear_square(Square.E2)
        self.board.set_piece_at(Square.E4, Piece.PAWN, Color.WHITE)
        
        # Center pawn should get bonus
        score = self.evaluator.evaluate(self.board)
        self.assertGreater(score, 0)


class TestSearch(unittest.TestCase):
    """Test search algorithms."""
    
    def setUp(self):
        """Set up test components."""
        self.board = ChessBoard()
        self.move_generator = MoveGenerator()
        self.evaluator = Evaluator()
        self.search_engine = SearchEngine(self.evaluator, self.move_generator)
    
    def test_transposition_table(self):
        """Test transposition table functionality."""
        tt = TranspositionTable(1)  # 1MB table
        
        # Store and retrieve entry
        tt.store(12345, 5, 100, TranspositionTable.EXACT)
        entry = tt.probe(12345)
        
        self.assertIsNotNone(entry)
        self.assertEqual(entry.depth, 5)
        self.assertEqual(entry.score, 100)
        self.assertEqual(entry.flag, TranspositionTable.EXACT)
    
    def test_basic_search(self):
        """Test basic search functionality."""
        # Search from starting position
        best_move, score, info = self.search_engine.search(self.board, depth=3, time_limit=1.0)
        
        # Should find a reasonable move
        self.assertIsNotNone(best_move)
        self.assertIsInstance(score, (int, float))
        self.assertGreater(info.nodes, 0)
    
    def test_mate_detection(self):
        """Test mate detection in search."""
        # Set up a simple mate position
        # This would require specific position setup
        pass  # TODO: Implement specific mate position test


class TestZobristHashing(unittest.TestCase):
    """Test Zobrist hashing functionality."""
    
    def setUp(self):
        """Set up test components."""
        self.hasher = ZobristHasher()
        self.board = ChessBoard()
    
    def test_position_hashing(self):
        """Test position hash calculation."""
        hash1 = self.hasher.hash_position(self.board)
        
        # Make a move
        move = Move(Square.E2, Square.E4, Piece.PAWN)
        self.board.make_move(move)
        hash2 = self.hasher.hash_position(self.board)
        
        # Hashes should be different
        self.assertNotEqual(hash1, hash2)
    
    def test_hash_consistency(self):
        """Test hash consistency across identical positions."""
        hash1 = self.hasher.hash_position(self.board)
        
        # Create identical board
        board2 = ChessBoard()
        hash2 = self.hasher.hash_position(board2)
        
        self.assertEqual(hash1, hash2)


class TestPerformance(unittest.TestCase):
    """Performance tests for critical components."""
    
    def setUp(self):
        """Set up test components."""
        self.board = ChessBoard()
        self.move_generator = MoveGenerator()
    
    def test_move_generation_speed(self):
        """Test move generation performance."""
        import time
        
        start_time = time.time()
        for _ in range(1000):
            moves = list(self.move_generator.generate_legal_moves(self.board))
        end_time = time.time()
        
        # Should generate moves quickly
        total_time = end_time - start_time
        self.assertLess(total_time, 1.0)  # Less than 1 second for 1000 iterations
        
        print(f"Move generation: {total_time:.3f}s for 1000 iterations")


def run_perft_test(board: ChessBoard, depth: int, move_generator: MoveGenerator) -> int:
    """
    Performance test (Perft) for move generation validation.
    Counts the number of leaf nodes at a given depth.
    """
    if depth == 0:
        return 1
    
    count = 0
    for move in move_generator.generate_legal_moves(board):
        board.make_move(move)
        count += run_perft_test(board, depth - 1, move_generator)
        # board.unmake_move()  # Would need proper implementation
    
    return count


class TestPerft(unittest.TestCase):
    """Perft tests for move generation validation."""
    
    def setUp(self):
        """Set up test components."""
        self.board = ChessBoard()
        self.move_generator = MoveGenerator()
    
    def test_perft_depth_1(self):
        """Test Perft depth 1 from starting position."""
        # Starting position should have 20 legal moves
        moves = list(self.move_generator.generate_legal_moves(self.board))
        self.assertEqual(len(moves), 20)
    
    # TODO: Add more Perft tests with known node counts


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestBitBoard,
        TestChessBoard,
        TestMoveGeneration,
        TestEvaluation,
        TestSearch,
        TestZobristHashing,
        TestPerformance,
        TestPerft
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\nTests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    # Exit with error code if tests failed
    sys.exit(0 if result.wasSuccessful() else 1)
