// Chess AI Integration
class ChessAI {
    constructor() {
        this.apiUrl = 'http://localhost:5000'; // Consolidated backend URL
        this.difficulty = 5; // Default depth
        this.thinking = false;
        this.backendAvailable = false;
        this.currentEngine = 'demo';
        this.availableEngines = {};
        this.gameId = 'default';
        this.checkBackend();
    }

    async checkBackend() {
        // Check if consolidated backend is available
        try {
            const response = await fetch(`${this.apiUrl}/api/health`);
            if (response.ok) {
                const data = await response.json();
                this.backendAvailable = true;
                this.availableEngines = data.server_info?.available_engines || {};
                this.currentEngine = data.server_info?.current_engine || 'demo';
                console.log('✅ Consolidated backend connected');
                console.log('Available engines:', this.availableEngines);
            }
        } catch (e) {
            this.backendAvailable = false;
            console.log('⚠️ Backend not available, using fallback AI');
        }
    }

    getCurrentApiUrl() {
        return this.apiUrl;
    }

    async makeMove(board, difficulty = null) {
        if (this.thinking) return null;
        
        this.thinking = true;
        const searchDepth = difficulty || this.difficulty;
        
        try {
            // Try to use backend AI first
            const backendMove = await this.getBackendMove(board, searchDepth);
            if (backendMove) {
                this.thinking = false;
                return backendMove;
            }
            
            // Fallback to local AI
            const legalMoves = this.getAllLegalMovesForCurrentPlayer(board);
            
            if (legalMoves.length === 0) {
                this.thinking = false;
                return null;
            }

            // Simulate thinking time
            await this.delay(500 + Math.random() * 1500);
            
            // Simple AI: prefer captures, avoid hanging pieces
            const move = this.selectBestMove(board, legalMoves, searchDepth);
            
            this.thinking = false;
            return move;
            
        } catch (error) {
            console.error('AI Error:', error);
            this.thinking = false;
            return null;
        }
    }

    getAllLegalMovesForCurrentPlayer(board) {
        const moves = [];
        const currentPlayer = board.currentPlayer;
        
        for (let row = 0; row < 8; row++) {
            for (let col = 0; col < 8; col++) {
                const piece = board.getPieceAt(row, col);
                if (piece) {
                    const pieceColor = piece[0] === 'w' ? 'white' : 'black';
                    if (pieceColor === currentPlayer) {
                        const pieceMoves = board.getLegalMoves(row, col);
                        pieceMoves.forEach(move => {
                            moves.push({
                                from: { row, col },
                                to: { row: move.row, col: move.col },
                                type: move.type,
                                piece: piece
                            });
                        });
                    }
                }
            }
        }
        
        return moves;
    }

    selectBestMove(board, legalMoves, depth) {
        // Simple evaluation-based move selection
        let bestMove = null;
        let bestScore = -Infinity;

        for (const move of legalMoves) {
            const score = this.evaluateMove(board, move, depth);
            if (score > bestScore) {
                bestScore = score;
                bestMove = move;
            }
        }

        return bestMove || legalMoves[Math.floor(Math.random() * legalMoves.length)];
    }

    evaluateMove(board, move, depth) {
        let score = 0;

        // Piece values
        const pieceValues = {
            'p': 100, 'n': 300, 'b': 300, 'r': 500, 'q': 900, 'k': 10000
        };

        // Capture bonus
        const targetPiece = board.getPieceAt(move.to.row, move.to.col);
        if (targetPiece) {
            score += pieceValues[targetPiece[1]] * 10; // High reward for captures
        }

        // Center control bonus
        const centerSquares = [[3,3], [3,4], [4,3], [4,4]];
        const extendedCenter = [[2,2], [2,3], [2,4], [2,5], [3,2], [3,5], [4,2], [4,5], [5,2], [5,3], [5,4], [5,5]];
        
        if (centerSquares.some(([r, c]) => r === move.to.row && c === move.to.col)) {
            score += 50;
        } else if (extendedCenter.some(([r, c]) => r === move.to.row && c === move.to.col)) {
            score += 20;
        }

        // Development bonus (moving pieces from back rank)
        if (move.piece[1] !== 'p' && move.piece[1] !== 'k') {
            const backRank = move.piece[0] === 'w' ? 7 : 0;
            if (move.from.row === backRank && move.to.row !== backRank) {
                score += 30;
            }
        }

        // Castling bonus
        if (move.type === 'castle-kingside' || move.type === 'castle-queenside') {
            score += 60;
        }

        // Check bonus
        const originalPiece = board.getPieceAt(move.to.row, move.to.col);
        board.setPieceAt(move.to.row, move.to.col, move.piece);
        board.setPieceAt(move.from.row, move.from.col, null);
        
        const enemyColor = board.currentPlayer === 'white' ? 'black' : 'white';
        if (board.isInCheck(enemyColor)) {
            score += 100;
        }

        // Restore board
        board.setPieceAt(move.from.row, move.from.col, move.piece);
        board.setPieceAt(move.to.row, move.to.col, originalPiece);

        // Random factor for variety
        score += Math.random() * 10;

        return score;
    }

    // Minimax with alpha-beta pruning (simplified)
    minimax(board, depth, alpha, beta, maximizingPlayer) {
        if (depth === 0) {
            return this.evaluatePosition(board);
        }

        const moves = this.getAllLegalMovesForCurrentPlayer(board);
        
        if (moves.length === 0) {
            if (board.isInCheck(board.currentPlayer)) {
                return maximizingPlayer ? -10000 : 10000; // Checkmate
            }
            return 0; // Stalemate
        }

        if (maximizingPlayer) {
            let maxEval = -Infinity;
            for (const move of moves) {
                // Make move
                const captured = board.getPieceAt(move.to.row, move.to.col);
                board.setPieceAt(move.to.row, move.to.col, move.piece);
                board.setPieceAt(move.from.row, move.from.col, null);
                board.currentPlayer = board.currentPlayer === 'white' ? 'black' : 'white';                const evaluation = this.minimax(board, depth - 1, alpha, beta, false);
                
                // Unmake move
                board.setPieceAt(move.from.row, move.from.col, move.piece);
                board.setPieceAt(move.to.row, move.to.col, captured);
                board.currentPlayer = board.currentPlayer === 'white' ? 'black' : 'white';

                maxEval = Math.max(maxEval, evaluation);
                alpha = Math.max(alpha, evaluation);
                if (beta <= alpha) break; // Alpha-beta pruning
            }
            return maxEval;
        } else {
            let minEval = Infinity;
            for (const move of moves) {
                // Make move
                const captured = board.getPieceAt(move.to.row, move.to.col);
                board.setPieceAt(move.to.row, move.to.col, move.piece);
                board.setPieceAt(move.from.row, move.from.col, null);
                board.currentPlayer = board.currentPlayer === 'white' ? 'black' : 'white';                const evaluation = this.minimax(board, depth - 1, alpha, beta, true);
                
                // Unmake move
                board.setPieceAt(move.from.row, move.from.col, move.piece);
                board.setPieceAt(move.to.row, move.to.col, captured);
                board.currentPlayer = board.currentPlayer === 'white' ? 'black' : 'white';

                minEval = Math.min(minEval, evaluation);
                beta = Math.min(beta, evaluation);
                if (beta <= alpha) break; // Alpha-beta pruning
            }
            return minEval;
        }
    }

    evaluatePosition(board) {
        let score = 0;
        const pieceValues = {
            'p': 100, 'n': 300, 'b': 300, 'r': 500, 'q': 900, 'k': 10000
        };

        // Material evaluation
        for (let row = 0; row < 8; row++) {
            for (let col = 0; col < 8; col++) {
                const piece = board.getPieceAt(row, col);
                if (piece) {
                    const value = pieceValues[piece[1]];
                    if (piece[0] === 'w') {
                        score += value;
                    } else {
                        score -= value;
                    }
                }
            }
        }

        // Positional factors
        score += this.evaluatePosition_mobility(board);
        score += this.evaluatePosition_safety(board);

        return board.currentPlayer === 'white' ? score : -score;
    }

    evaluatePosition_mobility(board) {
        const whiteMoves = board.getAllLegalMoves('white').length;
        const blackMoves = board.getAllLegalMoves('black').length;
        return (whiteMoves - blackMoves) * 10;
    }

    evaluatePosition_safety(board) {
        let score = 0;
        
        // King safety
        const whiteKing = board.findKing('white');
        const blackKing = board.findKing('black');
        
        if (whiteKing && board.isInCheck('white')) {
            score -= 100;
        }
        if (blackKing && board.isInCheck('black')) {
            score += 100;
        }

        return score;
    }

    async callPythonBackend(fen, depth) {
        try {
            const response = await fetch(`${this.apiUrl}/get_best_move`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    fen: fen,
                    depth: depth
                })
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            return await response.json();
        } catch (error) {
            console.warn('Python backend not available, using JavaScript AI');
            return null;
        }
    }

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    setDifficulty(depth) {
        this.difficulty = Math.max(1, Math.min(10, depth));
    }

    isThinking() {
        return this.thinking;
    }    async getBackendMove(board, depth) {
        try {
            if (!this.backendAvailable) return null;
            
            const fen = board.toFEN ? board.toFEN() : 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
            
            // Use new consolidated API
            const response = await fetch(`${this.apiUrl}/api/make_move`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    game_id: this.gameId,
                    move_data: {
                        fen: fen,
                        depth: depth,
                        engine: this.currentEngine
                    }
                })
            });
            
            if (!response.ok) {
                console.log('Backend not available, using local AI');
                return null;
            }
            
            const data = await response.json();
            console.log('Backend AI response:', data);
            
            if (data.ai_move) {
                // Update analysis display
                if (window.game && window.game.ui) {
                    window.game.ui.updateAnalysis({
                        evaluation: data.evaluation,
                        depth: data.depth,
                        nodes: data.nodes || 0,
                        time: data.thinking_time,
                        bestMove: data.ai_move,
                        engine: data.engine_used
                    });
                }
                
                return {
                    from: data.ai_move.from,
                    to: data.ai_move.to,
                    type: data.ai_move.type || 'normal',
                    promotion: data.ai_move.promotion
                };
            }
            
            return null;
        } catch (error) {
            console.log('Backend error:', error);
            return null;
        }
    }

    // Method to set AI engine
    async setEngine(engineName) {
        if (!this.backendAvailable) return false;
        
        try {
            const response = await fetch(`${this.apiUrl}/api/set_engine`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    engine: engineName
                })
            });
            
            if (response.ok) {
                const data = await response.json();
                if (data.success) {
                    this.currentEngine = data.current_engine;
                    console.log(`✅ Switched to engine: ${this.currentEngine}`);
                    return true;
                }
            }
        } catch (error) {
            console.error('Error setting engine:', error);
        }
        
        return false;
    }

    // Method to get available engines
    async getEngines() {
        if (!this.backendAvailable) return {};
        
        try {
            const response = await fetch(`${this.apiUrl}/api/engines`);
            if (response.ok) {
                const data = await response.json();
                return data.engines || {};
            }
        } catch (error) {
            console.error('Error getting engines:', error);
        }
        
        return {};
    }
}

// Python Flask Backend Integration (optional)
class PythonChessBackend {
    constructor() {
        this.baseUrl = 'http://localhost:5000';
        this.available = false;
        this.checkAvailability();
    }

    async checkAvailability() {
        try {
            const response = await fetch(`${this.baseUrl}/health`, {
                method: 'GET',
                timeout: 2000
            });
            this.available = response.ok;
        } catch (error) {
            this.available = false;
        }
    }

    async getBestMove(fen, depth = 5) {
        if (!this.available) return null;

        try {
            const response = await fetch(`${this.baseUrl}/get_best_move`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ fen, depth })
            });

            return await response.json();
        } catch (error) {
            console.error('Backend error:', error);
            return null;
        }
    }

    async evaluatePosition(fen) {
        if (!this.available) return null;

        try {
            const response = await fetch(`${this.baseUrl}/evaluate`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ fen })
            });

            return await response.json();
        } catch (error) {
            console.error('Backend error:', error);
            return null;
        }
    }
}
