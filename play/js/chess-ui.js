// Unified Chess UI with Complete Functionality
class ChessUI {
    constructor(chessEngine) {
        this.engine = chessEngine;
        this.selectedSquare = null;
        this.possibleMoves = [];
        this.boardElement = null;
        this.onMoveCompleted = null;
        this.boardFlipped = false;
        
        // Enhanced properties
        this.moveHistory = [];
        this.currentMoveIndex = -1;
        this.gameStates = []; // Store board states for each move
        this.boardTheme = 'classic';
        this.customThemeColors = {
            light: '#ece8d9',
            dark: '#b8860b',
            border: '#2c2c2c',
            pieceWhite: '#f5f5f5',
            pieceBlack: '#1a1a1a'
        };
        
        // Initialize captured pieces tracking
        this.capturedPieces = {
            white: [],
            black: []
        };
        
        // Piece symbols for display
        this.pieceSymbols = {
            'wp': '♙', 'wr': '♖', 'wn': '♘', 'wb': '♗', 'wq': '♕', 'wk': '♔',
            'bp': '♟', 'br': '♜', 'bn': '♞', 'bb': '♝', 'bq': '♛', 'bk': '♚'
        };
        
        this.promotionCallback = null;
        
        this.initializeBoard();
        this.bindEvents();
        this.initializeEnhancedFeatures();
        this.bindEnhancedEvents();
        this.updateDisplay();
    }
    
    initializeBoard() {
        this.boardElement = document.getElementById('chess-board');
        if (!this.boardElement) {
            console.error('Chess board element not found');
            return;
        }
        
        this.createBoard();
    }
    
    createBoard() {
        this.boardElement.innerHTML = '';
        
        for (let row = 0; row < 8; row++) {
            for (let col = 0; col < 8; col++) {
                const square = document.createElement('div');
                square.className = `square ${(row + col) % 2 === 0 ? 'light' : 'dark'}`;
                square.dataset.row = row;
                square.dataset.col = col;
                
                // Add piece element
                const piece = document.createElement('div');
                piece.className = 'piece';
                square.appendChild(piece);
                
                this.boardElement.appendChild(square);
            }
        }
    }
    
    initializeEnhancedFeatures() {
        // Store initial game state
        if (this.engine && typeof this.engine.exportState === 'function') {
            this.gameStates = [this.engine.exportState()];
        }
        this.updateMoveControls();
        this.applyTheme('classic');
        
        // Update displays
        this.updateCapturedPieces();
        this.updateCurrentPlayerDisplay();
    }
    
    bindEvents() {
        if (!this.boardElement) return;
        
        this.boardElement.addEventListener('click', (e) => {
            const square = e.target.closest('.square');
            if (square) {
                const row = parseInt(square.dataset.row);
                const col = parseInt(square.dataset.col);
                this.handleSquareClick(row, col);
            }
        });
    }
      bindEnhancedEvents() {
        // Theme selector
        const themeSelector = document.getElementById('board-theme');
        if (themeSelector) {
            themeSelector.addEventListener('change', (e) => {
                this.applyTheme(e.target.value);
            });
        }
        
        // Game control buttons
        const flipBoardBtn = document.getElementById('flip-board-btn');
        if (flipBoardBtn) {
            flipBoardBtn.addEventListener('click', () => {
                this.flipBoard();
            });
        }
        
        const undoBtn = document.getElementById('undo-btn');
        if (undoBtn) {
            undoBtn.addEventListener('click', () => {
                this.undoMove();
            });
        }
        
        // Player selection boxes
        const player1Box = document.getElementById('player1-box');
        const player2Box = document.getElementById('player2-box');
        
        if (player1Box) {
            player1Box.addEventListener('click', () => {
                this.togglePlayerDropdown('player1');
            });
        }
        
        if (player2Box) {
            player2Box.addEventListener('click', () => {
                this.togglePlayerDropdown('player2');
            });
        }
        
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
        const moveFirst = document.getElementById('move-first');
        const movePrev = document.getElementById('move-prev');
        const moveNext = document.getElementById('move-next');
        const moveLast = document.getElementById('move-last');
        
        if (moveFirst) moveFirst.addEventListener('click', () => this.goToMove(-1));
        if (movePrev) movePrev.addEventListener('click', () => this.goToMove(this.currentMoveIndex - 1));
        if (moveNext) moveNext.addEventListener('click', () => this.goToMove(this.currentMoveIndex + 1));
        if (moveLast) moveLast.addEventListener('click', () => this.goToMove(this.moveHistory.length - 1));
        
        // Export PGN button
        const exportBtn = document.getElementById('export-pgn');
        if (exportBtn) {
            exportBtn.addEventListener('click', () => {
                this.downloadPGN();
            });
        }
        
        // Move clicks for navigation
        const movesContainer = document.getElementById('moves-container');
        if (movesContainer) {
            movesContainer.addEventListener('click', (e) => {
                if (e.target.classList.contains('move')) {
                    const moveIndex = parseInt(e.target.dataset.moveIndex);
                    this.goToMove(moveIndex);
                }
            });
        }
        
        // Promotion dialog events
        const promotionPieces = document.querySelectorAll('.promotion-piece');
        promotionPieces.forEach(piece => {
            piece.addEventListener('click', (e) => {
                const pieceType = e.target.dataset.piece;
                this.handlePromotion(pieceType);
            });
        });
    }
    
    handleSquareClick(row, col) {
        if (this.selectedSquare) {
            // Try to make a move
            const fromRow = this.selectedSquare.row;
            const fromCol = this.selectedSquare.col;
            
            if (fromRow === row && fromCol === col) {
                // Clicked same square, deselect
                this.clearSelection();
                return;
            }
            
            // Check if this is a valid move
            if (this.isValidMove(fromRow, fromCol, row, col)) {
                this.makeMove(fromRow, fromCol, row, col);
            } else {
                // Invalid move, select new square if it has a piece
                this.selectSquare(row, col);
            }
        } else {
            // Select square if it has a piece
            this.selectSquare(row, col);
        }
    }
    
    selectSquare(row, col) {
        const piece = this.engine.getPieceAt(row, col);
        if (piece && piece[0] === this.engine.currentPlayer[0]) {
            this.selectedSquare = { row, col };
            this.possibleMoves = this.engine.getLegalMoves(row, col);
            this.updateDisplay();
        }
    }
    
    clearSelection() {
        this.selectedSquare = null;
        this.possibleMoves = [];
        this.updateDisplay();
    }
    
    isValidMove(fromRow, fromCol, toRow, toCol) {
        return this.possibleMoves.some(move => 
            move.toRow === toRow && move.toCol === toCol
        );
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
            this.updateDisplay();
            
            if (this.onMoveCompleted) {
                this.onMoveCompleted();
            }
        }
        
        return success;
    }
    
    recordMove(moveData) {
        // If we're not at the end of the move history, truncate it
        if (this.currentMoveIndex < this.moveHistory.length - 1) {
            this.moveHistory = this.moveHistory.slice(0, this.currentMoveIndex + 1);
            this.gameStates = this.gameStates.slice(0, this.currentMoveIndex + 2);
        }
        
        // Add the new move
        this.moveHistory.push(moveData);
        if (this.engine && typeof this.engine.exportState === 'function') {
            this.gameStates.push(this.engine.exportState());
        }
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
    
    updateDisplay() {
        this.updateBoard();
        this.updateCurrentPlayer();
        this.updateCapturedPieces();
        this.updateCurrentPlayerDisplay();
    }
    
    updateBoard() {
        if (!this.boardElement) return;
        
        const squares = this.boardElement.querySelectorAll('.square');
        
        squares.forEach(square => {
            const row = parseInt(square.dataset.row);
            const col = parseInt(square.dataset.col);
            const piece = this.engine.getPieceAt(row, col);
            const pieceElement = square.querySelector('.piece');
            
            // Update piece display
            if (piece) {
                // Set text content as fallback
                pieceElement.textContent = this.pieceSymbols[piece] || '';
                // Set class for piece type and color, plus image class
                pieceElement.className = `piece ${piece[0] === 'w' ? 'white' : 'black'} ${piece}`;
                
                // Try to load image and handle fallback
                const testImg = new Image();
                testImg.onload = () => {
                    pieceElement.classList.add('has-image');
                    pieceElement.classList.remove('no-image');
                };
                testImg.onerror = () => {
                    pieceElement.classList.add('no-image');
                    pieceElement.classList.remove('has-image');
                };
                testImg.src = `assets/${piece}.png`;
            } else {
                pieceElement.textContent = '';
                pieceElement.className = 'piece';
            }
            
            // Update square highlighting
            square.classList.remove('selected', 'legal-move', 'capture-move', 'last-move', 'possible-move');
            
            // Highlight selected square
            if (this.selectedSquare && this.selectedSquare.row === row && this.selectedSquare.col === col) {
                square.classList.add('selected');
            }
            
            // Highlight possible moves
            if (this.possibleMoves.some(move => move.toRow === row && move.toCol === col)) {
                square.classList.add('possible-move');
            }
        });
        
        this.updateAllPieceColors();
    }
    
    updateCurrentPlayer() {
        // Basic implementation - can be enhanced
    }
    
    showAIThinking(show) {
        const thinkingElement = document.getElementById('ai-thinking');
        if (thinkingElement) {
            thinkingElement.style.display = show ? 'flex' : 'none';
        }
    }
    
    updateAIStats(depth, nodes, time) {
        const depthElement = document.getElementById('search-depth');
        const nodesElement = document.getElementById('nodes-evaluated');
        const timeElement = document.getElementById('time-taken');
        
        if (depthElement) depthElement.textContent = depth;
        if (nodesElement) nodesElement.textContent = nodes.toLocaleString();
        if (timeElement) timeElement.textContent = time.toFixed(2) + 's';
    }
    
    // Theme Management
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
            if (piece.textContent) {
                piece.classList.toggle('white', isWhite);
                piece.classList.toggle('black', !isWhite);
            }
        });
    }
    
    // Move History Management
    updateMoveHistory() {
        const container = document.getElementById('moves-container');
        if (!container) return;
        
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
        
        if (firstBtn) firstBtn.disabled = this.currentMoveIndex <= -1;
        if (prevBtn) prevBtn.disabled = this.currentMoveIndex <= -1;
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
            if (this.engine && typeof this.engine.importState === 'function') {
                this.engine.importState(this.gameStates[moveIndex + 1]);
            }
        } else {
            // Go to initial position
            if (this.engine && typeof this.engine.importState === 'function') {
                this.engine.importState(this.gameStates[0]);
            }
        }
        
        this.clearSelection();
        this.updateBoard();
        this.updateDisplay();
        this.updateMoveHistory();
        this.updateMoveControls();
    }
    
    // Promotion Dialog
    showPromotionDialog(fromRow, fromCol, toRow, toCol, moveData) {
        const modal = document.getElementById('promotion-dialog');
        if (!modal) return;
        
        modal.style.display = 'flex';
        
        this.promotionCallback = (promotionPiece) => {
            moveData.promotion = promotionPiece;
            moveData.notation += '=' + promotionPiece.toUpperCase();
            
            const success = this.engine.makeMove(fromRow, fromCol, toRow, toCol, promotionPiece);
            
            if (success) {
                this.recordMove(moveData);
                this.clearSelection();
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
    
    // Captured Pieces Management
    updateCapturedPieces() {
        // Reset captured pieces arrays
        this.capturedPieces = { white: [], black: [] };
        
        // Get all pieces currently on the board
        const boardPieces = {};
        for (let row = 0; row < 8; row++) {
            for (let col = 0; col < 8; col++) {
                const piece = this.engine.getPieceAt(row, col);
                if (piece) {
                    boardPieces[piece] = (boardPieces[piece] || 0) + 1;
                }
            }
        }
        
        // Compare with starting position to find captured pieces
        const startingPieces = {
            white: ['wp', 'wp', 'wp', 'wp', 'wp', 'wp', 'wp', 'wp', 'wr', 'wr', 'wn', 'wn', 'wb', 'wb', 'wq', 'wk'],
            black: ['bp', 'bp', 'bp', 'bp', 'bp', 'bp', 'bp', 'bp', 'br', 'br', 'bn', 'bn', 'bb', 'bb', 'bq', 'bk']
        };
        
        // Find captured pieces by comparing counts
        ['white', 'black'].forEach(color => {
            startingPieces[color].forEach(piece => {
                const onBoard = boardPieces[piece] || 0;
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
    
    // Player Management
    togglePlayerDropdown(playerId) {
        const dropdown = document.getElementById(`${playerId}-dropdown`);
        if (dropdown) {
            dropdown.style.display = dropdown.style.display === 'block' ? 'none' : 'block';
        }
        
        // Close other dropdowns
        ['player1', 'player2'].forEach(id => {
            if (id !== playerId) {
                const otherDropdown = document.getElementById(`${id}-dropdown`);
                if (otherDropdown) {
                    otherDropdown.style.display = 'none';
                }
            }
        });
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
        if (dropdown) {
            dropdown.style.display = 'none';
        }
        
        // Update game mode based on selections
        this.updateGameModeFromPlayers();
    }
    
    updateGameModeFromPlayers() {
        const player1Label = document.querySelector('#player1-box .player-label');
        const player2Label = document.querySelector('#player2-box .player-label');
        
        if (!player1Label || !player2Label) return;
        
        const player1 = player1Label.textContent.toLowerCase();
        const player2 = player2Label.textContent.toLowerCase();
        
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
            if (typeof window.chessGame.updateGameModeDisplay === 'function') {
                window.chessGame.updateGameModeDisplay();
            }
            if (typeof window.chessGame.updateUIForGameMode === 'function') {
                window.chessGame.updateUIForGameMode();
            }
        }
    }
    
    // Enhanced current player display
    updateCurrentPlayerDisplay() {
        const currentTurnElement = document.getElementById('current-turn-text');
        
        if (currentTurnElement) {
            const isWhiteTurn = this.engine.currentPlayer === 'white';
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
    
    // PGN Export/Import
    exportPGN() {
        let pgn = '[Event "Chess AI Game"]\n';
        pgn += '[Site "Chess AI Interface"]\n';
        pgn += '[Date "' + new Date().toISOString().split('T')[0] + '"]\n';
        pgn += '[White "Human"]\n';
        pgn += '[Black "Chess AI"]\n';
        pgn += '[Result "*"]\n\n';
        
        for (let i = 0; i < this.moveHistory.length; i += 2) {
            const moveNum = Math.floor(i / 2) + 1;
            pgn += `${moveNum}. `;
            
            if (this.moveHistory[i]) {
                pgn += this.moveHistory[i].notation + ' ';
            }
            
            if (this.moveHistory[i + 1]) {
                pgn += this.moveHistory[i + 1].notation + ' ';
            }
            
            if (i % 10 === 8) pgn += '\n'; // Line break every 5 moves
        }
        
        return pgn.trim();
    }
    
    downloadPGN() {
        const pgn = this.exportPGN();
        const blob = new Blob([pgn], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `chess-game-${new Date().toISOString().split('T')[0]}.pgn`;
        a.click();
        URL.revokeObjectURL(url);
    }
    
    importPGN(pgn) {
        // Basic PGN parser - more sophisticated implementation could be added
        try {
            const moves = pgn.split('\n').filter(line => !line.startsWith('[') && line.trim())
                            .join(' ').split(/\d+\./).filter(move => move.trim());
            
            this.resetGame();
            
            for (const moveStr of moves) {
                const moveParts = moveStr.trim().split(/\s+/);
                for (const move of moveParts) {
                    if (move && !move.includes('*') && !move.includes('1-0') && 
                        !move.includes('0-1') && !move.includes('1/2-1/2')) {
                        // This would require implementing algebraic notation parsing
                        console.log('Parsing move:', move);
                    }
                }
            }
        } catch (error) {
            console.error('Error importing PGN:', error);
        }
    }
    
    // Board Controls
    flipBoard() {
        this.boardFlipped = !this.boardFlipped;
        
        // Re-create board with flipped perspective
        this.boardElement.innerHTML = '';
        
        for (let row = 0; row < 8; row++) {
            for (let col = 0; col < 8; col++) {
                const displayRow = this.boardFlipped ? 7 - row : row;
                const displayCol = this.boardFlipped ? 7 - col : col;
                
                const square = document.createElement('div');
                square.className = `square ${(displayRow + displayCol) % 2 === 0 ? 'light' : 'dark'}`;
                square.dataset.row = row;
                square.dataset.col = col;
                
                // Add piece element
                const piece = document.createElement('div');
                piece.className = 'piece';
                square.appendChild(piece);
                
                this.boardElement.appendChild(square);
            }
        }
        
        this.updateDisplay();
    }
    
    undoMove() {
        if (this.currentMoveIndex >= 0) {
            this.goToMove(this.currentMoveIndex - 1);
        }
    }
    
    resetGame() {
        this.clearSelection();
        this.moveHistory = [];
        this.currentMoveIndex = -1;
        if (this.engine && typeof this.engine.exportState === 'function') {
            this.gameStates = [this.engine.exportState()];
        }
        this.updateMoveHistory();
        this.updateMoveControls();
        this.updateDisplay();
    }
    
    // Analysis features
    updateAnalysis(analysisData) {
        if (!analysisData) return;
        
        const evaluationElement = document.getElementById('evaluation-score');
        const depthElement = document.getElementById('search-depth');
        const nodesElement = document.getElementById('nodes-evaluated');
        const timeElement = document.getElementById('time-taken');
        const bestMoveElement = document.getElementById('best-move');
        const engineElement = document.getElementById('engine-used');
        
        if (evaluationElement && analysisData.evaluation !== undefined) {
            evaluationElement.textContent = analysisData.evaluation.toFixed(2);
        }
        if (depthElement && analysisData.depth) {
            depthElement.textContent = analysisData.depth;
        }
        if (nodesElement && analysisData.nodes) {
            nodesElement.textContent = analysisData.nodes.toLocaleString();
        }
        if (timeElement && analysisData.time) {
            timeElement.textContent = analysisData.time.toFixed(2) + 's';
        }
        if (bestMoveElement && analysisData.bestMove) {
            bestMoveElement.textContent = this.formatMoveForDisplay(analysisData.bestMove);
        }
        if (engineElement && analysisData.engine) {
            engineElement.textContent = analysisData.engine;
        }
    }
    
    formatMoveForDisplay(move) {
        if (!move) return '';
        
        const files = 'abcdefgh';
        const ranks = '87654321';
        
        if (move.from && move.to) {
            const fromSquare = files[move.from.col] + ranks[move.from.row];
            const toSquare = files[move.to.col] + ranks[move.to.row];
            return fromSquare + toSquare + (move.promotion ? move.promotion : '');
        }
        
        return move.toString();
    }
    
    formatMove(move) {
        // Format move for PGN notation
        if (!move) return '';
        
        if (typeof move === 'string') return move;
        
        return this.formatMoveForDisplay(move);
    }
    
    updatePlayerDisplay() {
        // Update player color displays if needed
        const humanColorElement = document.getElementById('human-color');
        if (humanColorElement && window.chessGame) {
            humanColorElement.textContent = window.chessGame.playerColor === 'white' ? 'White' : 'Black';
        }
    }
}

// Export for use in other modules
if (typeof window !== 'undefined') {
    window.ChessUI = ChessUI;
}
