// Enhanced Chess UI with Theme Support and Move Recording
class EnhancedChessUI extends ChessUI {
    constructor(chessEngine) {
        // Initialize enhanced properties BEFORE calling super
        this.moveHistory = [];
        this.currentMoveIndex = -1;
        this.gameStates = []; // Store board states for each move
        this.customThemeColors = {
            light: '#ece8d9',
            dark: '#b8860b',
            border: '#2c2c2c',
            pieceWhite: '#f5f5f5',
            pieceBlack: '#1a1a1a'
        };
        
        super(chessEngine);
        
        this.initializeEnhancedFeatures();
        this.bindEnhancedEvents();
    }
      initializeEnhancedFeatures() {
        // Store initial game state
        this.gameStates = [this.engine.exportState()];
        this.updateMoveControls();
        this.applyTheme('classic');
        
        // Initialize captured pieces tracking
        this.capturedPieces = {
            white: [],
            black: []
        };
        this.updateCapturedPieces();
        this.updateCurrentPlayerDisplay();
    }
    
    bindEnhancedEvents() {
        // Theme selector
        document.getElementById('board-theme').addEventListener('change', (e) => {
            this.applyTheme(e.target.value);
        });
        
        // Player selection boxes
        document.getElementById('player1-box').addEventListener('click', () => {
            this.togglePlayerDropdown('player1');
        });
        
        document.getElementById('player2-box').addEventListener('click', () => {
            this.togglePlayerDropdown('player2');
        });
        
        // Player selection changes
        document.querySelectorAll('.player-select').forEach(select => {
            select.addEventListener('change', (e) => {
                this.handlePlayerChange(e.target);
            });
        });

        // Custom color inputs
        const colorInputs = ['custom-light', 'custom-dark', 'custom-border', 'custom-piece-white', 'custom-piece-black'];
        colorInputs.forEach(id => {
            const input = document.getElementById(id);
            if (input) {
                input.addEventListener('change', (e) => {
                    this.updateCustomColor(id, e.target.value);
                });
            }
        });
        
        // Move navigation buttons
        document.getElementById('move-first').addEventListener('click', () => this.goToMove(0));
        document.getElementById('move-prev').addEventListener('click', () => this.goToMove(this.currentMoveIndex - 1));
        document.getElementById('move-next').addEventListener('click', () => this.goToMove(this.currentMoveIndex + 1));
        document.getElementById('move-last').addEventListener('click', () => this.goToMove(this.moveHistory.length));
        
        // Move clicks for navigation
        document.getElementById('moves-container').addEventListener('click', (e) => {
            if (e.target.classList.contains('move')) {
                const moveIndex = parseInt(e.target.dataset.moveIndex);
                this.goToMove(moveIndex + 1);
            }
        });
    }
    
    applyTheme(themeName) {
        this.boardTheme = themeName;
        const body = document.body;
        
        // Remove all existing theme classes
        body.classList.remove('theme-classic', 'theme-modern', 'theme-minimal', 'theme-wood', 
                            'theme-neon', 'theme-ocean', 'theme-forest', 'theme-sunset', 'theme-custom');
        
        // Add new theme class
        body.classList.add(`theme-${themeName}`);
        
        // Show/hide custom color controls
        const customControls = document.getElementById('custom-theme-controls');
        if (customControls) {
            customControls.style.display = themeName === 'custom' ? 'block' : 'none';
        }
        
        // Update pieces with theme colors
        this.updateAllPieceColors();
    }
    
    updateCustomColor(colorType, color) {
        const colorMap = {
            'custom-light': '--custom-light',
            'custom-dark': '--custom-dark',
            'custom-border': '--custom-border',
            'custom-piece-white': '--custom-piece-white',
            'custom-piece-black': '--custom-piece-black'
        };
        
        if (colorMap[colorType]) {
            // Convert hex to HSL
            const hsl = this.hexToHsl(color);
            document.documentElement.style.setProperty(colorMap[colorType], hsl);
            this.updateAllPieceColors();
        }
    }
    
    hexToHsl(hex) {
        const r = parseInt(hex.slice(1, 3), 16) / 255;
        const g = parseInt(hex.slice(3, 5), 16) / 255;
        const b = parseInt(hex.slice(5, 7), 16) / 255;
        
        const max = Math.max(r, g, b);
        const min = Math.min(r, g, b);
        let h, s, l = (max + min) / 2;
        
        if (max === min) {
            h = s = 0;
        } else {
            const d = max - min;
            s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
            switch (max) {
                case r: h = (g - b) / d + (g < b ? 6 : 0); break;
                case g: h = (b - r) / d + 2; break;
                case b: h = (r - g) / d + 4; break;
            }
            h /= 6;
        }
        
        return `${Math.round(h * 360)} ${Math.round(s * 100)}% ${Math.round(l * 100)}%`;
    }
    
    updateAllPieceColors() {
        document.querySelectorAll('.piece').forEach(piece => {
            const isWhite = piece.textContent && ['♔', '♕', '♖', '♗', '♘', '♙'].includes(piece.textContent);
            piece.className = `piece ${isWhite ? 'white' : 'black'}`;
        });
    }
    
    makeMove(fromRow, fromCol, toRow, toCol) {
        // Store current state before making move
        const moveData = {
            from: { row: fromRow, col: fromCol },
            to: { row: toRow, col: toCol },
            piece: this.engine.getPieceAt(fromRow, fromCol),
            capturedPiece: this.engine.getPieceAt(toRow, toCol),
            notation: this.generateMoveNotation(fromRow, fromCol, toRow, toCol),
            timestamp: Date.now()
        };
        
        // Check for promotion
        const piece = this.engine.getPieceAt(fromRow, fromCol);
        if (piece && piece[1] === 'p' && (toRow === 0 || toRow === 7)) {
            this.showPromotionDialog(fromRow, fromCol, toRow, toCol, moveData);
            return;
        }
        
        // Make the move
        const success = this.engine.makeMove(fromRow, fromCol, toRow, toCol);
        
        if (success) {
            this.recordMove(moveData);
            this.clearSelection();
            this.updateBoard();
            this.updateDisplay();
            
            if (this.onMoveCompleted) {
                this.onMoveCompleted();
            }
        }
    }
    
    recordMove(moveData) {
        // If we're not at the end of the move history, truncate it
        if (this.currentMoveIndex < this.moveHistory.length - 1) {
            this.moveHistory = this.moveHistory.slice(0, this.currentMoveIndex + 1);
            this.gameStates = this.gameStates.slice(0, this.currentMoveIndex + 2);
        }
        
        // Add the new move
        this.moveHistory.push(moveData);
        this.gameStates.push(this.engine.exportState());
        this.currentMoveIndex = this.moveHistory.length - 1;
        
        this.updateMoveHistory();
        this.updateMoveControls();
    }
    
    generateMoveNotation(fromRow, fromCol, toRow, toCol) {
        const files = 'abcdefgh';
        const ranks = '87654321';
        
        const fromSquare = files[fromCol] + ranks[fromRow];
        const toSquare = files[toCol] + ranks[toRow];
        
        const piece = this.engine.getPieceAt(fromRow, fromCol);
        const capturedPiece = this.engine.getPieceAt(toRow, toCol);
        
        let notation = '';
        
        // Add piece letter (except for pawns)
        if (piece && piece[1] !== 'p') {
            notation += piece[1].toUpperCase();
        }
        
        // Add capture notation
        if (capturedPiece) {
            if (piece && piece[1] === 'p') {
                notation += files[fromCol];
            }
            notation += 'x';
        }
        
        notation += toSquare;
        
        return notation;
    }
    
    updateMoveHistory() {
        const container = document.getElementById('moves-container');
        container.innerHTML = '';
        
        for (let i = 0; i < this.moveHistory.length; i += 2) {
            const movePair = document.createElement('div');
            movePair.className = 'move-pair';
            
            const moveNumber = document.createElement('span');
            moveNumber.className = 'move-number';
            moveNumber.textContent = `${Math.floor(i / 2) + 1}.`;
            movePair.appendChild(moveNumber);
            
            // White move
            if (this.moveHistory[i]) {
                const whiteMove = document.createElement('span');
                whiteMove.className = `move white-move ${i === this.currentMoveIndex ? 'current' : ''}`;
                whiteMove.textContent = this.moveHistory[i].notation;
                whiteMove.dataset.moveIndex = i;
                whiteMove.title = `Move ${i + 1}: ${this.moveHistory[i].notation}`;
                movePair.appendChild(whiteMove);
            }
            
            // Black move
            if (this.moveHistory[i + 1]) {
                const blackMove = document.createElement('span');
                blackMove.className = `move black-move ${i + 1 === this.currentMoveIndex ? 'current' : ''}`;
                blackMove.textContent = this.moveHistory[i + 1].notation;
                blackMove.dataset.moveIndex = i + 1;
                blackMove.title = `Move ${i + 2}: ${this.moveHistory[i + 1].notation}`;
                movePair.appendChild(blackMove);
            }
            
            container.appendChild(movePair);
        }
        
        // Auto-scroll to latest move
        container.scrollTop = container.scrollHeight;
    }
    
    updateMoveControls() {
        const firstBtn = document.getElementById('move-first');
        const prevBtn = document.getElementById('move-prev');
        const nextBtn = document.getElementById('move-next');
        const lastBtn = document.getElementById('move-last');
        
        if (firstBtn) firstBtn.disabled = this.currentMoveIndex <= 0;
        if (prevBtn) prevBtn.disabled = this.currentMoveIndex <= 0;
        if (nextBtn) nextBtn.disabled = this.currentMoveIndex >= this.moveHistory.length - 1;
        if (lastBtn) lastBtn.disabled = this.currentMoveIndex >= this.moveHistory.length - 1;
    }
    
    goToMove(moveIndex) {
        // Clamp moveIndex to valid range
        moveIndex = Math.max(-1, Math.min(moveIndex, this.moveHistory.length - 1));
        
        if (moveIndex === this.currentMoveIndex) return;
        
        this.currentMoveIndex = moveIndex;
        
        // Restore game state
        if (moveIndex >= 0 && moveIndex < this.gameStates.length - 1) {
            this.engine.importState(this.gameStates[moveIndex + 1]);
        } else {
            // Go to initial position
            this.engine.importState(this.gameStates[0]);
        }
        
        this.clearSelection();
        this.updateBoard();
        this.updateDisplay();
        this.updateMoveHistory();
        this.updateMoveControls();
    }
    
    resetGame() {
        super.resetGame();
        this.moveHistory = [];
        this.currentMoveIndex = -1;
        this.gameStates = [this.engine.exportState()];
        this.updateMoveHistory();
        this.updateMoveControls();
    }
    
    updateBoard() {
        super.updateBoard();
        this.updateAllPieceColors();
    }
    
    showPromotionDialog(fromRow, fromCol, toRow, toCol, moveData) {
        const modal = document.getElementById('promotion-modal');
        modal.style.display = 'flex';
        
        this.promotionCallback = (promotionPiece) => {
            moveData.promotion = promotionPiece;
            moveData.notation += '=' + promotionPiece.toUpperCase();
            
            const success = this.engine.makeMove(fromRow, fromCol, toRow, toCol, promotionPiece);
            
            if (success) {
                this.recordMove(moveData);
                this.clearSelection();
                this.updateBoard();
                this.updateDisplay();
                
                if (this.onMoveCompleted) {
                    this.onMoveCompleted();
                }
            }
            
            modal.style.display = 'none';
            this.promotionCallback = null;
        };
    }
    
    handlePromotion(pieceType) {
        if (this.promotionCallback) {
            this.promotionCallback(pieceType);
        }
    }
    
    // Export game as PGN
    exportPGN() {
        let pgn = '[Event "Chess AI Game"]\\n';
        pgn += '[Site "Chess AI Interface"]\\n';
        pgn += '[Date "' + new Date().toISOString().split('T')[0] + '"]\\n';
        pgn += '[White "Human"]\\n';
        pgn += '[Black "Chess AI"]\\n';
        pgn += '[Result "*"]\\n\\n';
        
        for (let i = 0; i < this.moveHistory.length; i += 2) {
            const moveNum = Math.floor(i / 2) + 1;
            pgn += `${moveNum}. `;
            
            if (this.moveHistory[i]) {
                pgn += this.moveHistory[i].notation + ' ';
            }
            
            if (this.moveHistory[i + 1]) {
                pgn += this.moveHistory[i + 1].notation + ' ';
            }
            
            if (i % 10 === 8) pgn += '\\n'; // Line break every 5 moves
        }
        
        return pgn.trim();
    }
    
    // Import game from PGN (basic implementation)
    importPGN(pgn) {
        // This would require a full PGN parser - simplified for now
        console.log('PGN import functionality would be implemented here');
    }
    
    // Captured pieces tracking
    updateCapturedPieces() {
        // Reset captured pieces arrays
        this.capturedPieces = { white: [], black: [] };
        
        // Get all pieces currently on the board
        const boardPieces = new Set();
        for (let row = 0; row < 8; row++) {
            for (let col = 0; col < 8; col++) {
                const piece = this.engine.getPieceAt(row, col);
                if (piece) {
                    boardPieces.add(piece);
                }
            }
        }
        
        // Compare with starting position to find captured pieces
        const startingPieces = {
            white: ['wp', 'wp', 'wp', 'wp', 'wp', 'wp', 'wp', 'wp', 'wr', 'wr', 'wn', 'wn', 'wb', 'wb', 'wq', 'wk'],
            black: ['bp', 'bp', 'bp', 'bp', 'bp', 'bp', 'bp', 'bp', 'br', 'br', 'bn', 'bn', 'bb', 'bb', 'bq', 'bk']
        };
        
        // Count pieces on board
        const pieceCounts = {};
        for (let row = 0; row < 8; row++) {
            for (let col = 0; col < 8; col++) {
                const piece = this.engine.getPieceAt(row, col);
                if (piece) {
                    pieceCounts[piece] = (pieceCounts[piece] || 0) + 1;
                }
            }
        }
        
        // Find captured pieces by comparing counts
        ['white', 'black'].forEach(color => {
            startingPieces[color].forEach(piece => {
                const onBoard = pieceCounts[piece] || 0;
                const expected = startingPieces[color].filter(p => p === piece).length;
                const captured = expected - onBoard;
                
                for (let i = 0; i < captured; i++) {
                    this.capturedPieces[color].push(piece);
                }
            });
        });
        
        // Update display
        this.displayCapturedPieces();
    }
    
    displayCapturedPieces() {
        ['white', 'black'].forEach(color => {
            const container = document.getElementById(`${color}-captured-pieces`);
            const countElement = document.getElementById(`${color}-captured-count`);
            
            if (container && countElement) {
                container.innerHTML = '';
                countElement.textContent = this.capturedPieces[color].length;
                
                this.capturedPieces[color].forEach(piece => {
                    const pieceElement = document.createElement('div');
                    pieceElement.className = 'captured-piece';
                    pieceElement.textContent = this.pieceSymbols[piece] || '?';
                    container.appendChild(pieceElement);
                });
            }
        });
    }
    
    // Player selection handlers
    togglePlayerDropdown(playerId) {
        const dropdown = document.getElementById(`${playerId}-dropdown`);
        if (dropdown) {
            dropdown.style.display = dropdown.style.display === 'block' ? 'none' : 'block';
        }
    }
    
    handlePlayerChange(selectElement) {
        const playerBox = selectElement.closest('.player-box');
        const playerLabel = playerBox.querySelector('.player-label');
        const newValue = selectElement.value.toUpperCase();
        
        playerLabel.textContent = newValue;
        
        // Update player box styling
        playerBox.className = `player-box ${newValue.toLowerCase()}-player`;
        
        // Hide dropdown after selection
        const dropdown = playerBox.querySelector('.player-dropdown');
        dropdown.style.display = 'none';
        
        // Update game mode based on selections
        this.updateGameModeFromPlayers();
    }
    
    updateGameModeFromPlayers() {
        const player1 = document.querySelector('#player1-box .player-label').textContent.toLowerCase();
        const player2 = document.querySelector('#player2-box .player-label').textContent.toLowerCase();
        
        let gameMode;
        if (player1 === 'human' && player2 === 'ai') {
            gameMode = 'human-vs-ai';
        } else if (player1 === 'ai' && player2 === 'ai') {
            gameMode = 'ai-vs-ai';
        } else if (player1 === 'human' && player2 === 'human') {
            gameMode = 'human-vs-human';
        } else {
            gameMode = 'ai-vs-human';
        }
        
        // Trigger game mode change
        if (window.chessGame) {
            window.chessGame.gameMode = gameMode;
            window.chessGame.updateGameModeDisplay();
            window.chessGame.updateUIForGameMode();
        }
    }
    
    // Enhanced current player display
    updateCurrentPlayerDisplay() {
        const currentTurnElement = document.getElementById('current-turn-text');
        const currentPlayerElement = document.getElementById('current-player');
        
        if (currentTurnElement) {
            const isWhiteTurn = this.engine.currentTurn === 'white';
            currentTurnElement.textContent = isWhiteTurn ? 'WHITE TO MOVE' : 'BLACK TO MOVE';
            
            // Update styling based on turn
            currentTurnElement.style.color = isWhiteTurn ? 'hsl(var(--foreground))' : 'hsl(var(--foreground))';
            currentTurnElement.style.background = isWhiteTurn ? 'hsl(var(--accent))' : 'hsl(var(--muted))';
        }
        
        // Update move counter
        this.updateMoveCounter();
    }
    
    updateMoveCounter() {
        const currentMoveElement = document.getElementById('current-move-number');
        const totalMovesElement = document.getElementById('total-moves');
        
        if (currentMoveElement && totalMovesElement) {
            currentMoveElement.textContent = this.currentMoveIndex + 1;
            totalMovesElement.textContent = this.moveHistory.length;
        }
    }
    
    // Override the updateDisplay method to include our enhancements
    updateDisplay() {
        super.updateDisplay();
        this.updateCapturedPieces();
        this.updateCurrentPlayerDisplay();
    }
    
    // Override makeMove to track captures
    makeMove(fromRow, fromCol, toRow, toCol) {
        // Store current state before making move
        const moveData = {
            from: { row: fromRow, col: fromCol },
            to: { row: toRow, col: toCol },
            piece: this.engine.getPieceAt(fromRow, fromCol),
            capturedPiece: this.engine.getPieceAt(toRow, toCol),
            notation: this.generateMoveNotation(fromRow, fromCol, toRow, toCol),
            timestamp: Date.now()
        };
        
        // Call parent makeMove
        const result = super.makeMove(fromRow, fromCol, toRow, toCol);
        
        if (result) {
            // Update captured pieces and current player display
            this.updateCapturedPieces();
            this.updateCurrentPlayerDisplay();
            
            // Record the move
            this.moveHistory.push(moveData);
            this.currentMoveIndex = this.moveHistory.length - 1;
            this.gameStates.push(this.engine.exportState());
            this.updateMoveHistory();
            this.updateMoveControls();
        }
        
        return result;
    }
}

// Replace the original ChessUI with the enhanced version
window.ChessUI = EnhancedChessUI;
