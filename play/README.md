# Chess Game Interface

## 🎮 Play Chess Against Your AI

This interactive chess game lets you play against your trained chess AI with a beautiful, modern web interface.

## 📁 Project Structure

```
play/
├── README.md                    # This file
├── index.html                   # Main game interface
├── chess_game_server.py         # Unified backend server
├── launch_game.py              # Game launcher script
├── start_game.bat              # Windows quick launcher
├── assets/                     # Chess piece images
│   ├── bb.png, bk.png, bn.png, bp.png, bq.png, br.png
│   └── wb.png, wk.png, wn.png, wp.png, wq.png, wr.png
├── css/                        # Stylesheets
│   ├── styles.css              # Main game styles
│   └── piece-images.css        # Chess piece styling
├── js/                         # JavaScript modules
│   ├── chess-ai.js             # AI integration and backend communication
│   ├── chess-logic.js          # Chess game logic and rules
│   ├── chess-ui.js             # User interface components
│   └── main.js                 # Main game controller
└── docs/                       # Documentation
    └── README.md               # Technical documentation
```

## 🚀 Quick Start

### Option 1: Windows Quick Start
```bash
# Double-click the batch file or run from command line
start_game.bat
```

### Option 2: Complete Game with AI Backend
```bash
# Start the complete game with AI backend
cd play
python launch_game.py
```

### Option 3: Backend Server Only
```bash
# Start just the backend server
cd play
python chess_game_server.py
```
Then open `http://localhost:5000` in your browser.

### Option 4: Frontend Only (No AI)
```bash
# Open index.html directly in your browser
cd play
# Open index.html in your web browser
```

## 🎯 Features

### ✨ Game Features
- **Beautiful Chess Board**: Modern, responsive design with multiple themes
- **Drag & Drop**: Intuitive piece movement with smooth animations
- **Legal Move Highlighting**: Visual indication of valid moves
- **Move History**: Complete game notation and navigation
- **Real-time Analysis**: Position evaluation and AI insights

### 🤖 AI Features
- **Neural Network Integration**: Uses your trained chess AI models
- **Adjustable Difficulty**: Multiple strength levels (1-10 depth)
- **Multiple AI Engines**: Support for different AI implementations
- **Performance Metrics**: Search statistics and evaluation data

### 🎨 Interface Features
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Multiple Themes**: Various board and piece styles
- **Customizable Settings**: Personalize your gaming experience
- **Accessibility**: Keyboard navigation and screen reader support

## 🎮 How to Play

### Basic Controls
1. **Select Piece**: Click on any of your pieces
2. **Move Piece**: Click destination square or drag & drop
3. **Special Moves**: Castling, en passant, and pawn promotion supported
4. **Game Controls**: New game, undo, settings via interface buttons

### Keyboard Shortcuts
- `Ctrl + N`: New Game
- `Ctrl + Z`: Undo Last Move
- `Ctrl + F`: Flip Board
- `Esc`: Cancel current selection

## ⚙️ Configuration

### AI Settings
- **Difficulty Level**: Adjust search depth and time limits
- **Engine Selection**: Choose between available AI engines
- **Analysis Mode**: Enable/disable real-time position analysis

### Interface Settings
- **Board Theme**: Select visual appearance
- **Piece Style**: Choose piece design
- **Animation Speed**: Control move animation timing
- **Sound Effects**: Toggle audio feedback

## 🔧 Development

### Architecture
- **Modular JavaScript**: Clean separation of concerns
- **Modern CSS**: Flexbox/Grid layouts with animations
- **RESTful API**: Clean backend communication
- **Progressive Enhancement**: Works without JavaScript fallbacks

### File Organization
- **CSS**: Organized in `css/` directory with modular stylesheets
- **JavaScript**: Modular components in `js/` directory
- **Assets**: All images and resources in `assets/` directory
- **Documentation**: Technical docs in `docs/` directory

### Integration Points
- **Neural Networks**: Connects to models in `../src/neural/`
- **Chess Logic**: Uses engines from `../src/core/`
- **Training Data**: Can load positions from `../training_data/`

## 🛠️ Technical Requirements

### Dependencies
- Python 3.8+
- Flask web framework
- Modern web browser (Chrome, Firefox, Safari, Edge)
- Trained chess AI models (from main project)

### Performance
- **Client-side**: Optimized JavaScript with efficient algorithms
- **Server-side**: Fast neural network inference
- **Network**: Minimal API calls with smart caching

## 🐛 Troubleshooting

### Common Issues

1. **Server won't start**
   - Check Python installation and dependencies
   - Verify port 5000 is available
   - Run: `pip install -r ../requirements.txt`

2. **AI not responding**
   - Ensure backend server is running
   - Check browser console for errors
   - Verify model files exist in `../models/`

3. **Interface issues**
   - Clear browser cache
   - Check file paths in `index.html`
   - Verify all assets are present

4. **Performance problems**
   - Reduce AI difficulty level
   - Close other browser tabs
   - Check system resources

## 📝 API Documentation

The backend server provides these endpoints:
- `GET /` - Serve the main game interface
- `POST /api/move` - Request AI move
- `GET /api/health` - Check server status
- `POST /api/analyze` - Analyze position
- `GET /api/engines` - List available AI engines
- `POST /api/reset` - Reset game state

## 🎨 Customization

### Adding Themes
1. Create new CSS rules in `css/styles.css`
2. Add theme selector in interface
3. Update theme switching logic in JavaScript

### Integrating New AI Engines
1. Modify `chess_game_server.py` to load new models
2. Update `js/chess-ai.js` for new engine communication
3. Add engine selection in game interface

### Custom Piece Sets
1. Add new piece images to `assets/` directory
2. Update `css/piece-images.css` with new styles
3. Add piece set selector to interface

## 🤝 Contributing

When adding new features:
1. Follow the existing code structure and naming conventions
2. Update this README if you add new files or change the structure
3. Test all game modes and browsers before committing
4. Document any new configuration options
5. Ensure mobile compatibility

## 📜 License

Part of the Chess-AI project. See the main project README for license information.

---

**Ready to challenge your AI? Start playing! 🎉**
