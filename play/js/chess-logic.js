// Chess Logic Engine
class ChessEngine {
    constructor() {
        this.board = this.createInitialBoard();
        this.currentPlayer = 'white';
        this.moveHistory = [];
        this.capturedPieces = { white: [], black: [] };
        this.gameState = 'playing'; // playing, check, checkmate, stalemate, draw
        this.enPassantTarget = null;
        this.castlingRights = {
            white: { kingside: true, queenside: true },
            black: { kingside: true, queenside: true }
        };
        this.halfMoveClock = 0;
        this.fullMoveNumber = 1;
    }

    createInitialBoard() {
        const board = Array(8).fill().map(() => Array(8).fill(null));
        
        // Place white pieces
        board[7] = ['wr', 'wn', 'wb', 'wq', 'wk', 'wb', 'wn', 'wr'];
        board[6] = Array(8).fill('wp');
        
        // Place black pieces
        board[0] = ['br', 'bn', 'bb', 'bq', 'bk', 'bb', 'bn', 'br'];
        board[1] = Array(8).fill('bp');
        
        return board;
    }

    resetGame() {
        this.board = this.createInitialBoard();
        this.currentPlayer = 'white';
        this.moveHistory = [];
        this.capturedPieces = { white: [], black: [] };
        this.gameState = 'playing';
        this.enPassantTarget = null;
        this.castlingRights = {
            white: { kingside: true, queenside: true },
            black: { kingside: true, queenside: true }
        };
        this.halfMoveClock = 0;
        this.fullMoveNumber = 1;
    }

    getPieceAt(row, col) {
        if (row < 0 || row > 7 || col < 0 || col > 7) return null;
        return this.board[row][col];
    }

    setPieceAt(row, col, piece) {
        if (row >= 0 && row <= 7 && col >= 0 && col <= 7) {
            this.board[row][col] = piece;
        }
    }

    isSquareEmpty(row, col) {
        return this.getPieceAt(row, col) === null;
    }

    isEnemyPiece(row, col, playerColor) {
        const piece = this.getPieceAt(row, col);
        if (!piece) return false;
        return piece[0] !== playerColor[0];
    }

    isFriendlyPiece(row, col, playerColor) {
        const piece = this.getPieceAt(row, col);
        if (!piece) return false;
        return piece[0] === playerColor[0];
    }

    getLegalMoves(row, col) {
        const piece = this.getPieceAt(row, col);
        if (!piece) return [];
        
        const pieceColor = piece[0] === 'w' ? 'white' : 'black';
        if (pieceColor !== this.currentPlayer) return [];

        const pieceType = piece[1];
        let moves = [];

        switch (pieceType) {
            case 'p':
                moves = this.getPawnMoves(row, col, pieceColor);
                break;
            case 'r':
                moves = this.getRookMoves(row, col, pieceColor);
                break;
            case 'n':
                moves = this.getKnightMoves(row, col, pieceColor);
                break;
            case 'b':
                moves = this.getBishopMoves(row, col, pieceColor);
                break;
            case 'q':
                moves = this.getQueenMoves(row, col, pieceColor);
                break;
            case 'k':
                moves = this.getKingMoves(row, col, pieceColor);
                break;
        }

        // Filter out moves that would put own king in check
        return moves.filter(move => !this.wouldBeInCheck(row, col, move.row, move.col, pieceColor));
    }

    getPawnMoves(row, col, color) {
        const moves = [];
        const direction = color === 'white' ? -1 : 1;
        const startRow = color === 'white' ? 6 : 1;

        // Forward move
        if (this.isSquareEmpty(row + direction, col)) {
            moves.push({ row: row + direction, col: col, type: 'normal' });
            
            // Double forward move from starting position
            if (row === startRow && this.isSquareEmpty(row + 2 * direction, col)) {
                moves.push({ row: row + 2 * direction, col: col, type: 'normal' });
            }
        }

        // Captures
        [-1, 1].forEach(colOffset => {
            const newRow = row + direction;
            const newCol = col + colOffset;
            
            if (this.isEnemyPiece(newRow, newCol, color)) {
                moves.push({ row: newRow, col: newCol, type: 'capture' });
            }
            
            // En passant
            if (this.enPassantTarget && 
                this.enPassantTarget.row === newRow && 
                this.enPassantTarget.col === newCol) {
                moves.push({ row: newRow, col: newCol, type: 'enpassant' });
            }
        });

        return moves;
    }

    getRookMoves(row, col, color) {
        const moves = [];
        const directions = [[0, 1], [0, -1], [1, 0], [-1, 0]];

        directions.forEach(([rowDir, colDir]) => {
            for (let i = 1; i < 8; i++) {
                const newRow = row + i * rowDir;
                const newCol = col + i * colDir;

                if (newRow < 0 || newRow > 7 || newCol < 0 || newCol > 7) break;

                if (this.isSquareEmpty(newRow, newCol)) {
                    moves.push({ row: newRow, col: newCol, type: 'normal' });
                } else if (this.isEnemyPiece(newRow, newCol, color)) {
                    moves.push({ row: newRow, col: newCol, type: 'capture' });
                    break;
                } else {
                    break; // Friendly piece blocks
                }
            }
        });

        return moves;
    }

    getKnightMoves(row, col, color) {
        const moves = [];
        const knightMoves = [
            [-2, -1], [-2, 1], [-1, -2], [-1, 2],
            [1, -2], [1, 2], [2, -1], [2, 1]
        ];

        knightMoves.forEach(([rowOffset, colOffset]) => {
            const newRow = row + rowOffset;
            const newCol = col + colOffset;

            if (newRow >= 0 && newRow <= 7 && newCol >= 0 && newCol <= 7) {
                if (this.isSquareEmpty(newRow, newCol)) {
                    moves.push({ row: newRow, col: newCol, type: 'normal' });
                } else if (this.isEnemyPiece(newRow, newCol, color)) {
                    moves.push({ row: newRow, col: newCol, type: 'capture' });
                }
            }
        });

        return moves;
    }

    getBishopMoves(row, col, color) {
        const moves = [];
        const directions = [[1, 1], [1, -1], [-1, 1], [-1, -1]];

        directions.forEach(([rowDir, colDir]) => {
            for (let i = 1; i < 8; i++) {
                const newRow = row + i * rowDir;
                const newCol = col + i * colDir;

                if (newRow < 0 || newRow > 7 || newCol < 0 || newCol > 7) break;

                if (this.isSquareEmpty(newRow, newCol)) {
                    moves.push({ row: newRow, col: newCol, type: 'normal' });
                } else if (this.isEnemyPiece(newRow, newCol, color)) {
                    moves.push({ row: newRow, col: newCol, type: 'capture' });
                    break;
                } else {
                    break; // Friendly piece blocks
                }
            }
        });

        return moves;
    }

    getQueenMoves(row, col, color) {
        return [...this.getRookMoves(row, col, color), ...this.getBishopMoves(row, col, color)];
    }

    getKingMoves(row, col, color) {
        const moves = [];
        const directions = [
            [-1, -1], [-1, 0], [-1, 1],
            [0, -1],           [0, 1],
            [1, -1],  [1, 0],  [1, 1]
        ];

        directions.forEach(([rowDir, colDir]) => {
            const newRow = row + rowDir;
            const newCol = col + colDir;

            if (newRow >= 0 && newRow <= 7 && newCol >= 0 && newCol <= 7) {
                if (this.isSquareEmpty(newRow, newCol)) {
                    moves.push({ row: newRow, col: newCol, type: 'normal' });
                } else if (this.isEnemyPiece(newRow, newCol, color)) {
                    moves.push({ row: newRow, col: newCol, type: 'capture' });
                }
            }
        });

        // Castling
        if (!this.isInCheck(color)) {
            // Kingside castling
            if (this.castlingRights[color].kingside &&
                this.isSquareEmpty(row, col + 1) &&
                this.isSquareEmpty(row, col + 2) &&
                !this.wouldBeInCheck(row, col, row, col + 1, color) &&
                !this.wouldBeInCheck(row, col, row, col + 2, color)) {
                moves.push({ row: row, col: col + 2, type: 'castle-kingside' });
            }

            // Queenside castling
            if (this.castlingRights[color].queenside &&
                this.isSquareEmpty(row, col - 1) &&
                this.isSquareEmpty(row, col - 2) &&
                this.isSquareEmpty(row, col - 3) &&
                !this.wouldBeInCheck(row, col, row, col - 1, color) &&
                !this.wouldBeInCheck(row, col, row, col - 2, color)) {
                moves.push({ row: row, col: col - 2, type: 'castle-queenside' });
            }
        }

        return moves;
    }

    findKing(color) {
        const kingPiece = color === 'white' ? 'wk' : 'bk';
        for (let row = 0; row < 8; row++) {
            for (let col = 0; col < 8; col++) {
                if (this.board[row][col] === kingPiece) {
                    return { row, col };
                }
            }
        }
        return null;
    }

    isInCheck(color) {
        const king = this.findKing(color);
        if (!king) return false;

        return this.isSquareAttacked(king.row, king.col, color === 'white' ? 'black' : 'white');
    }

    isSquareAttacked(row, col, byColor) {
        // Check all pieces of the attacking color
        for (let r = 0; r < 8; r++) {
            for (let c = 0; c < 8; c++) {
                const piece = this.board[r][c];
                if (!piece) continue;
                
                const pieceColor = piece[0] === 'w' ? 'white' : 'black';
                if (pieceColor !== byColor) continue;

                const moves = this.getPieceMoves(r, c, pieceColor, true); // true for attack calculation
                if (moves.some(move => move.row === row && move.col === col)) {
                    return true;
                }
            }
        }
        return false;
    }

    getPieceMoves(row, col, color, forAttack = false) {
        const piece = this.getPieceAt(row, col);
        if (!piece) return [];

        const pieceType = piece[1];
        switch (pieceType) {
            case 'p':
                return forAttack ? this.getPawnAttacks(row, col, color) : this.getPawnMoves(row, col, color);
            case 'r':
                return this.getRookMoves(row, col, color);
            case 'n':
                return this.getKnightMoves(row, col, color);
            case 'b':
                return this.getBishopMoves(row, col, color);
            case 'q':
                return this.getQueenMoves(row, col, color);
            case 'k':
                return this.getKingAttacks(row, col, color);
            default:
                return [];
        }
    }

    getPawnAttacks(row, col, color) {
        const moves = [];
        const direction = color === 'white' ? -1 : 1;

        [-1, 1].forEach(colOffset => {
            const newRow = row + direction;
            const newCol = col + colOffset;
            if (newRow >= 0 && newRow <= 7 && newCol >= 0 && newCol <= 7) {
                moves.push({ row: newRow, col: newCol, type: 'attack' });
            }
        });

        return moves;
    }

    getKingAttacks(row, col, color) {
        const moves = [];
        const directions = [
            [-1, -1], [-1, 0], [-1, 1],
            [0, -1],           [0, 1],
            [1, -1],  [1, 0],  [1, 1]
        ];

        directions.forEach(([rowDir, colDir]) => {
            const newRow = row + rowDir;
            const newCol = col + colDir;

            if (newRow >= 0 && newRow <= 7 && newCol >= 0 && newCol <= 7) {
                moves.push({ row: newRow, col: newCol, type: 'attack' });
            }
        });

        return moves;
    }

    wouldBeInCheck(fromRow, fromCol, toRow, toCol, color) {
        // Make the move temporarily
        const originalPiece = this.board[toRow][toCol];
        const movingPiece = this.board[fromRow][fromCol];
        
        this.board[toRow][toCol] = movingPiece;
        this.board[fromRow][fromCol] = null;

        const inCheck = this.isInCheck(color);

        // Restore the board
        this.board[fromRow][fromCol] = movingPiece;
        this.board[toRow][toCol] = originalPiece;

        return inCheck;
    }

    makeMove(fromRow, fromCol, toRow, toCol, promotionPiece = null) {
        const piece = this.getPieceAt(fromRow, fromCol);
        if (!piece) return false;

        const legalMoves = this.getLegalMoves(fromRow, fromCol);
        const targetMove = legalMoves.find(move => move.row === toRow && move.col === toCol);
        
        if (!targetMove) return false;

        // Store move for history
        const moveData = {
            from: { row: fromRow, col: fromCol },
            to: { row: toRow, col: toCol },
            piece: piece,
            captured: this.getPieceAt(toRow, toCol),
            type: targetMove.type,
            check: false,
            checkmate: false
        };

        // Handle captures
        if (targetMove.type === 'capture') {
            const capturedPiece = this.getPieceAt(toRow, toCol);
            const capturedColor = capturedPiece[0] === 'w' ? 'white' : 'black';
            this.capturedPieces[capturedColor].push(capturedPiece);
        }

        // Handle en passant
        if (targetMove.type === 'enpassant') {
            const capturedRow = this.currentPlayer === 'white' ? toRow + 1 : toRow - 1;
            const capturedPiece = this.getPieceAt(capturedRow, toCol);
            const capturedColor = capturedPiece[0] === 'w' ? 'white' : 'black';
            this.capturedPieces[capturedColor].push(capturedPiece);
            this.setPieceAt(capturedRow, toCol, null);
        }

        // Handle castling
        if (targetMove.type === 'castle-kingside') {
            // Move rook
            const rookRow = fromRow;
            this.setPieceAt(rookRow, 5, this.getPieceAt(rookRow, 7));
            this.setPieceAt(rookRow, 7, null);
        } else if (targetMove.type === 'castle-queenside') {
            // Move rook
            const rookRow = fromRow;
            this.setPieceAt(rookRow, 3, this.getPieceAt(rookRow, 0));
            this.setPieceAt(rookRow, 0, null);
        }

        // Make the move
        this.setPieceAt(toRow, toCol, piece);
        this.setPieceAt(fromRow, fromCol, null);

        // Handle pawn promotion
        if (piece[1] === 'p' && (toRow === 0 || toRow === 7)) {
            if (promotionPiece) {
                const color = piece[0];
                this.setPieceAt(toRow, toCol, color + promotionPiece);
                moveData.promotion = color + promotionPiece;
            }
        }

        // Update en passant target
        this.enPassantTarget = null;
        if (piece[1] === 'p' && Math.abs(toRow - fromRow) === 2) {
            this.enPassantTarget = { row: (fromRow + toRow) / 2, col: toCol };
        }

        // Update castling rights
        if (piece[1] === 'k') {
            this.castlingRights[this.currentPlayer].kingside = false;
            this.castlingRights[this.currentPlayer].queenside = false;
        } else if (piece[1] === 'r') {
            if (fromCol === 0) {
                this.castlingRights[this.currentPlayer].queenside = false;
            } else if (fromCol === 7) {
                this.castlingRights[this.currentPlayer].kingside = false;
            }
        }

        // Switch players
        this.currentPlayer = this.currentPlayer === 'white' ? 'black' : 'white';

        // Check for check/checkmate
        if (this.isInCheck(this.currentPlayer)) {
            moveData.check = true;
            if (this.isCheckmate(this.currentPlayer)) {
                moveData.checkmate = true;
                this.gameState = 'checkmate';
            } else {
                this.gameState = 'check';
            }
        } else if (this.isStalemate(this.currentPlayer)) {
            this.gameState = 'stalemate';
        } else {
            this.gameState = 'playing';
        }

        // Add to move history
        this.moveHistory.push(moveData);

        // Update move counters
        if (piece[1] === 'p' || moveData.captured) {
            this.halfMoveClock = 0;
        } else {
            this.halfMoveClock++;
        }

        if (this.currentPlayer === 'white') {
            this.fullMoveNumber++;
        }

        return true;
    }

    isCheckmate(color) {
        if (!this.isInCheck(color)) return false;
        return this.getAllLegalMoves(color).length === 0;
    }

    isStalemate(color) {
        if (this.isInCheck(color)) return false;
        return this.getAllLegalMoves(color).length === 0;
    }

    getAllLegalMoves(color) {
        const moves = [];
        for (let row = 0; row < 8; row++) {
            for (let col = 0; col < 8; col++) {
                const piece = this.getPieceAt(row, col);
                if (piece && ((piece[0] === 'w' && color === 'white') || (piece[0] === 'b' && color === 'black'))) {
                    const pieceMoves = this.getLegalMoves(row, col);
                    pieceMoves.forEach(move => {
                        moves.push({ from: { row, col }, to: move });
                    });
                }
            }
        }
        return moves;
    }

    undoLastMove() {
        if (this.moveHistory.length === 0) return false;

        const lastMove = this.moveHistory.pop();
        
        // Restore piece positions
        this.setPieceAt(lastMove.from.row, lastMove.from.col, lastMove.piece);
        this.setPieceAt(lastMove.to.row, lastMove.to.col, lastMove.captured);

        // Handle special moves
        if (lastMove.type === 'enpassant') {
            const capturedRow = this.currentPlayer === 'black' ? lastMove.to.row + 1 : lastMove.to.row - 1;
            this.setPieceAt(capturedRow, lastMove.to.col, lastMove.captured);
        } else if (lastMove.type === 'castle-kingside') {
            const rookRow = lastMove.from.row;
            this.setPieceAt(rookRow, 7, this.getPieceAt(rookRow, 5));
            this.setPieceAt(rookRow, 5, null);
        } else if (lastMove.type === 'castle-queenside') {
            const rookRow = lastMove.from.row;
            this.setPieceAt(rookRow, 0, this.getPieceAt(rookRow, 3));
            this.setPieceAt(rookRow, 3, null);
        }

        // Remove captured piece from captured list
        if (lastMove.captured) {
            const capturedColor = lastMove.captured[0] === 'w' ? 'white' : 'black';
            const index = this.capturedPieces[capturedColor].lastIndexOf(lastMove.captured);
            if (index > -1) {
                this.capturedPieces[capturedColor].splice(index, 1);
            }
        }

        // Switch player back
        this.currentPlayer = this.currentPlayer === 'white' ? 'black' : 'white';
        
        // Reset game state
        this.gameState = 'playing';

        return true;
    }

    toFEN() {
        let fen = '';
        
        // Board position
        for (let row = 0; row < 8; row++) {
            let emptyCount = 0;
            for (let col = 0; col < 8; col++) {
                const piece = this.board[row][col];
                if (piece) {
                    if (emptyCount > 0) {
                        fen += emptyCount;
                        emptyCount = 0;
                    }
                    fen += piece[1] === piece[1].toUpperCase() ? piece[1] : piece[1].toUpperCase();
                    if (piece[0] === 'b') {
                        fen = fen.slice(0, -1) + piece[1];
                    }
                } else {
                    emptyCount++;
                }
            }
            if (emptyCount > 0) {
                fen += emptyCount;
            }
            if (row < 7) fen += '/';
        }

        // Active color
        fen += ' ' + (this.currentPlayer === 'white' ? 'w' : 'b');

        // Castling rights
        fen += ' ';
        let castling = '';
        if (this.castlingRights.white.kingside) castling += 'K';
        if (this.castlingRights.white.queenside) castling += 'Q';
        if (this.castlingRights.black.kingside) castling += 'k';
        if (this.castlingRights.black.queenside) castling += 'q';
        fen += castling || '-';

        // En passant
        fen += ' ';
        if (this.enPassantTarget) {
            const file = String.fromCharCode(97 + this.enPassantTarget.col);
            const rank = 8 - this.enPassantTarget.row;
            fen += file + rank;
        } else {
            fen += '-';
        }

        // Halfmove clock and fullmove number
        fen += ` ${this.halfMoveClock} ${this.fullMoveNumber}`;        return fen;
    }
    
    // Export complete game state for move navigation
    exportState() {
        return {
            board: this.board.map(row => [...row]),
            currentPlayer: this.currentPlayer,
            capturedPieces: {
                white: [...this.capturedPieces.white],
                black: [...this.capturedPieces.black]
            },
            gameState: this.gameState,
            enPassantTarget: this.enPassantTarget ? { ...this.enPassantTarget } : null,
            castlingRights: {
                white: { ...this.castlingRights.white },
                black: { ...this.castlingRights.black }
            },
            halfMoveClock: this.halfMoveClock,
            fullMoveNumber: this.fullMoveNumber
        };
    }
    
    // Import complete game state for move navigation
    importState(state) {
        this.board = state.board.map(row => [...row]);
        this.currentPlayer = state.currentPlayer;
        this.capturedPieces = {
            white: [...state.capturedPieces.white],
            black: [...state.capturedPieces.black]
        };
        this.gameState = state.gameState;
        this.enPassantTarget = state.enPassantTarget ? { ...state.enPassantTarget } : null;
        this.castlingRights = {
            white: { ...state.castlingRights.white },
            black: { ...state.castlingRights.black }
        };
        this.halfMoveClock = state.halfMoveClock;
        this.fullMoveNumber = state.fullMoveNumber;
    }
    
    // Get current game position as FEN (for external engines)
    getCurrentPosition() {
        return this.toFEN();
    }
    
    // Set position from FEN
    setPosition(fen) {
        this.fromFEN(fen);
    }
}
