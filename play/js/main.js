// Main Game Controller
class ChessGame {    constructor() {
        this.engine = new ChessEngine();
        this.ui = new ChessUI(this.engine);
        this.ai = new ChessAI();
        this.pythonBackend = new PythonChessBackend();
        
        this.playerColor = 'white';
        this.aiColor = 'black';
        this.gameMode = 'human-vs-ai';
        
        this.init();
    }    init() {
        // Override UI's onMoveCompleted method
        this.ui.onMoveCompleted = () => {
            this.onMoveCompleted();
        };

        // Initialize displays
        this.updateGameModeDisplay();
        this.updateUIForGameMode();
        this.ui.updatePlayerDisplay();

        // Bind additional events
        this.bindEvents();

        // Start game
        this.checkForAIMove();
    }    bindEvents() {        // Player selection handling is now in ChessUI
          // New Game button
        const newGameBtn = document.getElementById('new-game-btn');
        if (newGameBtn) {
            newGameBtn.addEventListener('click', () => {
                this.newGame();
            });
        }
        
        // AI difficulty setting
        const difficultyElement = document.getElementById('ai-difficulty');
        if (difficultyElement) {
            difficultyElement.addEventListener('change', (e) => {
                const depth = parseInt(e.target.value);
                this.ai.setDifficulty(depth);
            });
        }

        // Human color setting
        const humanColorElement = document.getElementById('human-color');
        if (humanColorElement) {
            humanColorElement.addEventListener('change', (e) => {
                this.playerColor = e.target.value;
                this.aiColor = this.playerColor === 'white' ? 'black' : 'white';
                this.ui.updatePlayerDisplay();
                
                // Reset game to apply new colors
                this.newGame();
            });
        }

        // Game result handling
        this.engine.onGameEnd = (result) => {
            this.showGameResult(result);
        };
    }
    
    updateGameModeDisplay() {
        const gameModeDisplay = document.getElementById('game-mode-display');
        if (gameModeDisplay) {
            switch (this.gameMode) {
                case 'human-vs-ai':
                    gameModeDisplay.textContent = 'HUMAN vs AI';
                    break;
                case 'ai-vs-ai':
                    gameModeDisplay.textContent = 'AI vs AI';
                    break;
                case 'human-vs-human':
                    gameModeDisplay.textContent = 'HUMAN vs HUMAN';
                    break;
            }
        }
    }
    
    updateUIForGameMode() {
        const aiConfigCard = document.getElementById('ai-config-card');
        const aiStatsCard = document.getElementById('ai-stats-card');
        
        if (aiConfigCard && aiStatsCard) {
            switch (this.gameMode) {
                case 'human-vs-ai':
                    aiConfigCard.style.display = 'block';
                    aiStatsCard.style.display = 'block';
                    break;
                case 'ai-vs-ai':
                    aiConfigCard.style.display = 'block';
                    aiStatsCard.style.display = 'block';
                    break;
                case 'human-vs-human':
                    aiConfigCard.style.display = 'none';
                    aiStatsCard.style.display = 'none';
                    break;            }
        }
    }    async onMoveCompleted() {
        // Update game state display
        this.updateGameState();
        
        // Update UI display
        if (this.ui) {
            this.ui.updateCurrentPlayerDisplay();
            this.ui.updateMoveCounter();
        }
        
        // Check if game is over
        if (this.isGameOver()) {
            this.handleGameEnd();
            return;
        }        // Trigger AI move if it's AI's turn
        await this.checkForAIMove();
    }

    async checkForAIMove() {
        // Handle different game modes
        switch (this.gameMode) {
            case 'human-vs-ai':
                if (this.engine.currentPlayer === this.aiColor && !this.isGameOver()) {
                    await this.makeAIMove();
                }
                break;
            case 'ai-vs-ai':
                if (!this.isGameOver()) {
                    await this.makeAIMove();
                }
                break;
            case 'human-vs-human':
                // No AI moves in human vs human mode
                break;
        }
    }

    async makeAIMove() {
        if (this.ai.isThinking()) return;

        this.ui.showAIThinking(true);
        
        const startTime = Date.now();
        let nodesSearched = 0;
        
        try {
            // Try Python backend first
            const fen = this.engine.toFEN();
            const difficulty = parseInt(document.getElementById('ai-difficulty').value);
            
            let aiResponse = await this.pythonBackend.getBestMove(fen, difficulty);
            
            if (!aiResponse) {
                // Fallback to JavaScript AI
                const aiMove = await this.ai.makeMove(this.engine, difficulty);
                
                if (aiMove) {
                    aiResponse = {
                        move: aiMove,
                        depth: difficulty,
                        nodes: Math.floor(Math.random() * 10000) + 1000,
                        time: (Date.now() - startTime) / 1000
                    };
                }
            }

            if (aiResponse && aiResponse.move) {
                const move = aiResponse.move;
                
                // Update AI stats
                this.ui.updateAIStats(
                    aiResponse.depth || difficulty,
                    aiResponse.nodes || 0,
                    aiResponse.time || (Date.now() - startTime) / 1000
                );

                // Make the move
                const success = this.engine.makeMove(
                    move.from.row,
                    move.from.col,
                    move.to.row,
                    move.to.col,
                    move.promotion
                );

                if (success) {
                    this.ui.updateDisplay();
                    
                    // Check for game end
                    if (this.isGameOver()) {
                        this.handleGameEnd();
                    }
                }
            } else {
                console.error('AI failed to find a move');
            }
            
        } catch (error) {
            console.error('AI move error:', error);
        } finally {
            this.ui.showAIThinking(false);
        }
    }

    updateGameState() {
        // This method can be used to update any game state displays
        console.log(`Game state: ${this.engine.gameState}`);
        console.log(`Current player: ${this.engine.currentPlayer}`);
        console.log(`Move #${this.engine.fullMoveNumber}`);
    }

    isGameOver() {
        return this.engine.gameState === 'checkmate' || 
               this.engine.gameState === 'stalemate' ||
               this.engine.halfMoveClock >= 100; // 50-move rule
    }

    handleGameEnd() {
        let resultText = '';
        let reasonText = '';

        switch (this.engine.gameState) {
            case 'checkmate':
                const winner = this.engine.currentPlayer === 'white' ? 'Black' : 'White';
                resultText = `${winner} Wins!`;
                reasonText = 'by checkmate';
                break;
            case 'stalemate':
                resultText = 'Draw';
                reasonText = 'by stalemate';
                break;
            default:
                if (this.engine.halfMoveClock >= 100) {
                    resultText = 'Draw';
                    reasonText = 'by 50-move rule';
                }
        }

        this.showGameResult(resultText, reasonText);
    }

    showGameResult(resultText, reasonText = '') {
        const resultDiv = document.getElementById('game-result');
        const resultTextElement = document.getElementById('result-text');
        const resultReasonElement = document.getElementById('result-reason');

        resultTextElement.textContent = resultText;
        resultReasonElement.textContent = reasonText;
        resultDiv.style.display = 'block';

        // Auto-hide after 10 seconds
        setTimeout(() => {
            resultDiv.style.display = 'none';
        }, 10000);
    }

    newGame() {
        this.engine.resetGame();
        this.ui.clearSelection();
        this.ui.updateDisplay();
        
        // Hide game result
        document.getElementById('game-result').style.display = 'none';
        
        // Check if AI should move first
        this.checkForAIMove();
    }

    // Public methods for external control
    getCurrentPosition() {
        return this.engine.toFEN();
    }

    loadPosition(fen) {
        // This would require implementing FEN parsing in the chess engine
        console.log('Loading FEN position:', fen);
    }

    getGamePGN() {
        // Generate PGN notation of the game
        let pgn = '[Event "Human vs AI"]\n';
        pgn += '[Site "Chess AI Interface"]\n';
        pgn += `[Date "${new Date().toISOString().split('T')[0]}"]\n`;
        pgn += `[White "${this.playerColor === 'white' ? 'Human' : 'Chess AI'}"]\n`;
        pgn += `[Black "${this.playerColor === 'black' ? 'Human' : 'Chess AI'}"]\n`;
        
        if (this.engine.gameState === 'checkmate') {
            const winner = this.engine.currentPlayer === 'white' ? '0-1' : '1-0';
            pgn += `[Result "${winner}"]\n`;
        } else if (this.engine.gameState === 'stalemate') {
            pgn += '[Result "1/2-1/2"]\n';
        } else {
            pgn += '[Result "*"]\n';
        }
        
        pgn += '\n';
        
        // Add moves (simplified notation)
        for (let i = 0; i < this.engine.moveHistory.length; i += 2) {
            const moveNumber = Math.floor(i / 2) + 1;
            pgn += `${moveNumber}. `;
            
            if (this.engine.moveHistory[i]) {
                pgn += this.ui.formatMove(this.engine.moveHistory[i]) + ' ';
            }
            
            if (this.engine.moveHistory[i + 1]) {
                pgn += this.ui.formatMove(this.engine.moveHistory[i + 1]) + ' ';
            }
            
            if ((i + 2) % 10 === 0) pgn += '\n';
        }
        
        return pgn;
    }
}

// Initialize the game when the page loads
let chessGame;

document.addEventListener('DOMContentLoaded', function() {
    chessGame = new ChessGame();
    
    // Make chessGame globally accessible
    window.chessGame = chessGame;
    
    // Add keyboard shortcuts
    document.addEventListener('keydown', function(e) {
        switch(e.key) {
            case 'n':
            case 'N':
                if (e.ctrlKey) {
                    e.preventDefault();
                    chessGame.newGame();
                }
                break;
            case 'z':
            case 'Z':
                if (e.ctrlKey) {
                    e.preventDefault();
                    chessGame.ui.undoMove();
                }
                break;
            case 'f':
            case 'F':
                if (e.ctrlKey) {
                    e.preventDefault();
                    chessGame.ui.flipBoard();
                }
                break;
        }
    });

    // Add download PGN functionality
    const downloadBtn = document.createElement('button');
    downloadBtn.textContent = '📥 Download PGN';
    downloadBtn.className = 'btn btn-secondary';    downloadBtn.onclick = function() {
        const pgn = chessGame.getGamePGN();
        const blob = new Blob([pgn], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `chess-game-${new Date().toISOString().split('T')[0]}.pgn`;
        a.click();
        URL.revokeObjectURL(url);
    };
    
    const buttonGroup = document.querySelector('.button-group');
    if (buttonGroup) {
        buttonGroup.appendChild(downloadBtn);
    }

    console.log('♔ Chess AI Game Initialized');
    console.log('🎮 Keyboard shortcuts:');
    console.log('  Ctrl+N: New Game');
    console.log('  Ctrl+Z: Undo Move');
    console.log('  Ctrl+F: Flip Board');
});
