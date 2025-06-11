#!/usr/bin/env python3
"""
Chess AI Quick Launcher
======================
One-click launcher for training and playing your chess AI.
"""

import os
import sys
import subprocess
import webbrowser
import time
import argparse
from pathlib import Path

def print_banner():
    """Print the chess AI banner."""
    print("♔" * 60)
    print("    🚀 CHESS AI - QUICK LAUNCHER")
    print("♔" * 60)
    print()

def check_dependencies():
    """Check if required dependencies are installed."""
    print("🔍 Checking dependencies...")
    
    try:
        import torch
        print("✅ PyTorch available")
    except ImportError:
        print("❌ PyTorch not found. Install with: pip install torch")
        return False
    
    try:
        import flask
        print("✅ Flask available")
    except ImportError:
        print("❌ Flask not found. Install with: pip install flask flask-cors")
        return False
    
    return True

def quick_train():
    """Run quick training."""
    print("🧠 Starting quick AI training...")
    print("⏱️ This will take about 5-10 minutes")
    print()
    
    try:
        result = subprocess.run([sys.executable, 'train_ai.py', '--quick'], 
                              check=True, capture_output=False)
        print("\n✅ Training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed: {e}")
        return False

def start_game():
    """Start the chess game server."""
    print("🎮 Starting chess game server...")
    
    # Change to play directory
    play_dir = Path(__file__).parent / 'play'
    os.chdir(play_dir)
    
    # Start server in background
    try:
        print("📡 Server starting at http://localhost:5000")
        print("🌐 Opening browser...")
        
        # Start server
        process = subprocess.Popen([sys.executable, 'chess_game_server.py'])
        
        # Wait a moment for server to start
        time.sleep(3)
        
        # Open browser
        webbrowser.open('http://localhost:5000')
        
        print("🎯 Game is ready! Press Ctrl+C to stop the server.")
        
        # Wait for server
        process.wait()
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
        if 'process' in locals():
            process.terminate()
    except Exception as e:
        print(f"❌ Error starting game: {e}")

def main():
    """Main launcher function."""
    parser = argparse.ArgumentParser(description='Chess AI Quick Launcher')
    parser.add_argument('--train-only', action='store_true', help='Only train the AI')
    parser.add_argument('--play-only', action='store_true', help='Only start the game')
    parser.add_argument('--full-train', action='store_true', help='Run full production training')
    
    args = parser.parse_args()
    
    print_banner()
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Missing dependencies. Please install them and try again.")
        return
    
    print("✅ All dependencies available!")
    print()
    
    # Handle different modes
    if args.play_only:
        start_game()
        return
    
    if args.train_only:
        if args.full_train:
            print("🚀 Starting full production training...")
            subprocess.run([sys.executable, 'train_ai.py', '--full'])
        else:
            quick_train()
        return
    
    # Default: train then play
    print("🎯 Welcome to Chess AI Quick Launcher!")
    print()
    print("Options:")
    print("1. Quick train + play (recommended for first time)")
    print("2. Play with existing model")
    print("3. Full production training")
    print("4. Exit")
    print()
    
    while True:
        try:
            choice = input("Enter your choice (1-4): ").strip()
            
            if choice == '1':
                print("\n🚀 Starting quick training and game setup...")
                if quick_train():
                    print("\n🎮 Training complete! Starting game...")
                    time.sleep(2)
                    start_game()
                break
                
            elif choice == '2':
                print("\n🎮 Starting game with existing model...")
                start_game()
                break
                
            elif choice == '3':
                print("\n🏭 Starting full production training...")
                print("⚠️  This will take several hours!")
                confirm = input("Continue? (y/N): ").strip().lower()
                if confirm in ['y', 'yes']:
                    subprocess.run([sys.executable, 'train_ai.py', '--full'])
                break
                
            elif choice == '4':
                print("👋 Goodbye!")
                break
                
            else:
                print("❌ Invalid choice. Please enter 1-4.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except EOFError:
            print("\n\n👋 Goodbye!")
            break

if __name__ == "__main__":
    main()
