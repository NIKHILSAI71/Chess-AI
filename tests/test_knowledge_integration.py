#!/usr/bin/env python3
"""
Test script to verify knowledge integration components.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.board import ChessBoard, Move, Piece
from knowledge.opening_book import CustomOpeningBook, create_opening_book
from knowledge.tablebase import MockTablebase, TablebaseManager, create_tablebase

def test_opening_book():
    """Test opening book functionality."""
    print("Testing Opening Book Integration...")
    
    # Create a custom opening book
    book = create_opening_book(book_type='custom')
    
    # Test with starting position
    board = ChessBoard()    # Add some opening moves
    e2e4 = Move(12, 28, Piece.PAWN)  # e2-e4
    book.add_position(board, e2e4, weight=10)
    
    # Test book probe
    entries = book.probe(board)
    if entries:
        print(f"✓ Found {len(entries)} opening book entries")
        move = book.get_book_move(board)
        if move:
            print(f"✓ Book suggests move: {move}")
    else:
        print("✗ No opening book entries found")

def test_tablebase():
    """Test tablebase functionality."""
    print("\nTesting Tablebase Integration...")
    
    # Create tablebase manager
    manager = TablebaseManager()
    
    # Add mock tablebase
    mock_tb = create_tablebase(tablebase_type='mock')
    manager.add_prober(mock_tb)
    
    # Test with starting position
    board = ChessBoard()
    
    result, distance = manager.probe_position(board)
    print(f"✓ Tablebase probe result: {result}, distance: {distance}")
    
    # Test if position is in tablebase
    is_tb_pos = manager.is_tablebase_position(board)
    print(f"✓ Position in tablebase: {is_tb_pos}")

def test_enhanced_uci():
    """Test enhanced UCI engine."""
    print("\nTesting Enhanced UCI Engine...")
    
    try:
        from engine.uci import EnhancedUCIEngine
        
        engine = EnhancedUCIEngine()
        print(f"✓ Enhanced UCI engine created: {engine.name} v{engine.version}")
        print(f"✓ Neural network available: {hasattr(engine, 'neural_net') and engine.neural_net is not None}")
        print(f"✓ Hybrid engine available: {hasattr(engine, 'hybrid_engine') and engine.hybrid_engine is not None}")
        
    except Exception as e:
        print(f"✗ Enhanced UCI engine error: {e}")

def main():
    """Run all knowledge integration tests."""
    print("Chess-AI Knowledge Integration Test")
    print("=" * 40)
    
    try:
        test_opening_book()
        test_tablebase()
        test_enhanced_uci()
        
        print("\n" + "=" * 40)
        print("✓ Knowledge integration tests completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
