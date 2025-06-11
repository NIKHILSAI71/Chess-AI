#!/usr/bin/env python3
"""
Chess Game Launcher
==================
Launch the complete chess game with AI backend.
"""

import os
import sys
import subprocess
import webbrowser
import time
import threading

def start_backend():
    """Start the Python backend server."""
    print("🚀 Starting Chess AI Backend...")
    
    try:
        # Change to play directory
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
          # Start the Flask server
        subprocess.run([sys.executable, 'chess_game_server.py'], check=True)
        
    except KeyboardInterrupt:
        print("\n🛑 Backend server stopped")
    except Exception as e:
        print(f"❌ Error starting backend: {e}")

def open_web_interface():
    """Open the web interface in the default browser."""
    time.sleep(2)  # Wait for backend to start
    
    # Get the current directory and create file URL
    current_dir = os.path.dirname(os.path.abspath(__file__))
    html_file = os.path.join(current_dir, 'index.html')
    file_url = f"file:///{html_file.replace(os.sep, '/')}"
    
    print(f"🌐 Opening chess game at: {file_url}")
    webbrowser.open(file_url)

def main():
    print("♔ Chess AI Game Launcher")
    print("=" * 40)
      # Check if required files exist
    required_files = ['index.html', 'chess_game_server.py', 'css/styles.css']
    for file in required_files:
        if not os.path.exists(file):
            print(f"❌ Missing required file: {file}")
            return
    
    print("✅ All required files found")
    
    # Start web interface in a separate thread
    web_thread = threading.Thread(target=open_web_interface)
    web_thread.daemon = True
    web_thread.start()
    
    print("🎮 Starting chess game...")
    print("📝 Instructions:")
    print("   - Click pieces to select and move them")
    print("   - Use drag and drop to move pieces")
    print("   - Adjust AI difficulty in settings")
    print("   - Press Ctrl+C to stop the server")
    print()
    
    # Start backend (this will block)
    start_backend()

if __name__ == "__main__":
    main()
