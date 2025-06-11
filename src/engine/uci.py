"""
Enhanced UCI Protocol Implementation
===================================

Advanced UCI interface supporting neural networks, MCTS, and hybrid search modes.
Provides comprehensive engine configuration and analysis capabilities.
"""

import sys
import threading
import time
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Core imports
from core.board import ChessBoard, Move, Color
from core.moves import MoveGenerator
from core.evaluation import Evaluator

# Neural network imports
try:
    from neural.network import NeuralNetworkEvaluator
    from neural.hybrid_search import HybridSearchEngine, SearchMode, HybridSearchConfig
    NEURAL_AVAILABLE = True
except ImportError:
    NEURAL_AVAILABLE = False
    print("Neural network components not available")


@dataclass
class AdvancedEngineOptions:
    """Enhanced configuration options for the chess engine."""
    # Traditional options
    hash_size: int = 128              # MB for transposition table
    threads: int = 1                  # Number of search threads
    contempt: int = 0                 # Contempt factor
    move_overhead: int = 30           # Move overhead in ms
    
    # Neural network options
    neural_network_path: str = ""     # Path to neural network model
    use_neural_network: bool = False  # Enable neural network evaluation
    
    # Search mode options
    search_mode: str = "hybrid"       # classical, neural_mcts, hybrid, neural_ab
    mcts_simulations: int = 800       # MCTS simulations per move
    classical_depth: int = 12         # Classical search depth
    
    # Hybrid search options
    complexity_threshold: float = 0.5 # Complexity threshold for mode switching
    time_threshold: float = 5.0       # Time threshold for MCTS
    material_threshold: int = 1500    # Material threshold for endgame
    
    # Analysis options
    multi_pv: int = 1                 # Number of principal variations
    skill_level: int = 20             # Skill level (0-20)
    
    # Debug options
    debug_mode: bool = False          # Debug mode
    log_search: bool = False          # Log search information


class EnhancedUCIEngine:
    """
    Enhanced UCI protocol implementation with neural network support.
    
    Supports multiple search modes including classical alpha-beta, MCTS with neural networks,
    and adaptive hybrid approaches.
    """
    
    def __init__(self):
        """Initialize enhanced UCI engine."""
        # Core components
        self.board = ChessBoard()
        self.move_generator = MoveGenerator()
        self.evaluator = Evaluator()
        
        # Enhanced options
        self.options = AdvancedEngineOptions()
        
        # Search engines
        self.hybrid_engine = None
        self.neural_net = None
        self._initialize_search_engines()
        
        # Search state
        self.search_thread: Optional[threading.Thread] = None
        self.is_searching = False
        self.stop_search = False
        
        # Engine metadata
        self.name = "Chess-AI Enhanced"
        self.version = "2.0"
        self.author = "Chess-AI Development Team"
        
        # Performance tracking
        self.search_history = []
        self.analysis_cache = {}
    
    def _initialize_search_engines(self):
        """Initialize search engines based on available components."""
        try:
            if NEURAL_AVAILABLE:
                # Try to create hybrid engine
                from neural.hybrid_search import create_hybrid_engine
                self.hybrid_engine = create_hybrid_engine(
                    self.options.neural_network_path if self.options.neural_network_path else None
                )
                print("Initialized hybrid search engine")
            else:
                print("Neural components not available, using classical search only")
        except Exception as e:
            print(f"Failed to initialize neural components: {e}")
    
    def run(self):
        """Main UCI loop with enhanced features."""
        self._send(f"{self.name} v{self.version}")
        self._send("Enhanced UCI engine with neural network support")
        
        while True:
            try:
                line = input().strip()
                if line:
                    self._process_command(line)
            except EOFError:
                break
            except KeyboardInterrupt:
                break
    
    def _process_command(self, command: str):
        """Process UCI command with enhanced options."""
        parts = command.split()
        if not parts:
            return
        
        cmd = parts[0].lower()
        
        try:
            if cmd == "uci":
                self._handle_uci()
            elif cmd == "debug":
                self._handle_debug(parts)
            elif cmd == "isready":
                self._handle_isready()
            elif cmd == "setoption":
                self._handle_setoption(parts)
            elif cmd == "register":
                self._handle_register()
            elif cmd == "ucinewgame":
                self._handle_ucinewgame()
            elif cmd == "position":
                self._handle_position(parts)
            elif cmd == "go":
                self._handle_go(parts)
            elif cmd == "stop":
                self._handle_stop()
            elif cmd == "ponderhit":
                self._handle_ponderhit()
            elif cmd == "quit":
                self._handle_quit()
            # Enhanced commands
            elif cmd == "analyze":
                self._handle_analyze(parts)
            elif cmd == "eval":
                self._handle_eval()
            elif cmd == "perft":
                self._handle_perft(parts)
            elif cmd == "bench":
                self._handle_bench(parts)
            else:
                if self.options.debug_mode:
                    self._send(f"info string Unknown command: {command}")
        except Exception as e:
            if self.options.debug_mode:
                self._send(f"info string Error processing command '{command}': {e}")
    
    def _handle_uci(self):
        """Handle 'uci' command with enhanced options."""
        self._send(f"id name {self.name}")
        self._send(f"id author {self.author}")
        
        # Traditional UCI options
        self._send("option name Hash type spin default 128 min 1 max 8192")
        self._send("option name Threads type spin default 1 min 1 max 64")
        self._send("option name Contempt type spin default 0 min -200 max 200")
        self._send("option name Move Overhead type spin default 30 min 0 max 1000")
        
        # Enhanced options
        if NEURAL_AVAILABLE:
            self._send("option name Neural Network Path type string default")
            self._send("option name Use Neural Network type check default false")
            self._send("option name Search Mode type combo default hybrid var classical var neural_mcts var hybrid var neural_ab")
            self._send("option name MCTS Simulations type spin default 800 min 100 max 5000")
            self._send("option name Complexity Threshold type string default 0.5")
            self._send("option name Time Threshold type string default 5.0")
        
        self._send("option name Classical Depth type spin default 12 min 1 max 50")
        self._send("option name MultiPV type spin default 1 min 1 max 100")
        self._send("option name Skill Level type spin default 20 min 0 max 20")
        self._send("option name Debug Mode type check default false")
        self._send("option name Log Search type check default false")
        
        self._send("uciok")
    
    def _handle_setoption(self, parts: List[str]):
        """Handle 'setoption' command with enhanced options."""
        if len(parts) < 4 or parts[1].lower() != "name":
            return
        
        # Extract option name and value
        name_parts = []
        value_parts = []
        value_started = False
        
        for part in parts[2:]:
            if part.lower() == "value":
                value_started = True
            elif value_started:
                value_parts.append(part)
            else:
                name_parts.append(part)
        
        option_name = " ".join(name_parts).lower()
        option_value = " ".join(value_parts) if value_parts else ""
        
        try:
            # Handle traditional options
            if option_name == "hash":
                self.options.hash_size = int(option_value)
            elif option_name == "threads":
                self.options.threads = int(option_value)
            elif option_name == "contempt":
                self.options.contempt = int(option_value)
            elif option_name == "move overhead":
                self.options.move_overhead = int(option_value)
            
            # Handle enhanced options
            elif option_name == "neural network path":
                self.options.neural_network_path = option_value
                self._reload_neural_network()
            elif option_name == "use neural network":
                self.options.use_neural_network = option_value.lower() == "true"
            elif option_name == "search mode":
                self.options.search_mode = option_value.lower()
            elif option_name == "mcts simulations":
                self.options.mcts_simulations = int(option_value)
            elif option_name == "classical depth":
                self.options.classical_depth = int(option_value)
            elif option_name == "complexity threshold":
                self.options.complexity_threshold = float(option_value)
            elif option_name == "time threshold":
                self.options.time_threshold = float(option_value)
            elif option_name == "multipv":
                self.options.multi_pv = int(option_value)
            elif option_name == "skill level":
                self.options.skill_level = int(option_value)
            elif option_name == "debug mode":
                self.options.debug_mode = option_value.lower() == "true"
            elif option_name == "log search":
                self.options.log_search = option_value.lower() == "true"
            
            # Update search engine configuration
            self._update_search_config()
            
        except (ValueError, TypeError) as e:
            if self.options.debug_mode:
                self._send(f"info string Invalid option value: {e}")
    
    def _reload_neural_network(self):
        """Reload neural network with new path."""
        if not NEURAL_AVAILABLE or not self.options.neural_network_path:
            return
        
        try:
            self.neural_net = NeuralNetworkEvaluator(self.options.neural_network_path)
            # Reinitialize hybrid engine with new neural network
            from neural.hybrid_search import HybridSearchEngine, HybridSearchConfig
            config = HybridSearchConfig()
            self.hybrid_engine = HybridSearchEngine(
                self.move_generator, self.evaluator, self.neural_net, config
            )
            self._send("info string Neural network loaded successfully")
        except Exception as e:
            self._send(f"info string Failed to load neural network: {e}")
    
    def _update_search_config(self):
        """Update search engine configuration based on options."""
        if self.hybrid_engine:
            self.hybrid_engine.configure(
                max_depth=self.options.classical_depth,
                mcts_simulations=self.options.mcts_simulations,
                complexity_threshold=self.options.complexity_threshold,
                time_threshold=self.options.time_threshold
            )
    
    def _handle_go(self, parts: List[str]):
        """Handle 'go' command with enhanced search."""
        if self.is_searching:
            return
        
        # Parse go parameters
        params = self._parse_go_params(parts[1:])
        
        # Start search in separate thread
        self.stop_search = False
        self.search_thread = threading.Thread(
            target=self._search_worker, args=(params,)
        )
        self.search_thread.start()
    
    def _search_worker(self, params: Dict[str, Any]):
        """Enhanced search worker with multiple search modes."""
        self.is_searching = True
        start_time = time.time()
        
        try:
            # Determine search parameters
            time_limit = self._calculate_time_limit(params)
            depth = params.get('depth', self.options.classical_depth)
            
            # Select search mode
            search_mode = self._get_search_mode()
            
            if self.options.log_search:
                self._send(f"info string Using search mode: {search_mode}")
            
            # Execute search based on mode and availability
            if self.hybrid_engine and search_mode != "classical":
                best_move, search_info = self._search_with_hybrid_engine(
                    search_mode, time_limit, depth, params
                )
            else:
                best_move, search_info = self._search_classical(time_limit, depth)
            
            # Send results
            if not self.stop_search and best_move:
                # Send final info
                elapsed = time.time() - start_time
                self._send_search_info(search_info, elapsed)
                self._send(f"bestmove {best_move}")
            elif not self.stop_search:
                self._send("bestmove 0000")  # No move found
                
        except Exception as e:
            if self.options.debug_mode:
                self._send(f"info string Search error: {e}")
            self._send("bestmove 0000")
        finally:
            self.is_searching = False
    
    def _get_search_mode(self) -> str:
        """Get the current search mode."""
        if not NEURAL_AVAILABLE or not self.options.use_neural_network:
            return "classical"
        return self.options.search_mode
    
    def _search_with_hybrid_engine(self, mode: str, time_limit: float, 
                                  depth: int, params: Dict) -> tuple:
        """Search using the hybrid engine."""
        # Convert mode string to enum
        mode_map = {
            "classical": SearchMode.CLASSICAL,
            "neural_mcts": SearchMode.NEURAL_MCTS,
            "hybrid": SearchMode.HYBRID,
            "neural_ab": SearchMode.NEURAL_AB
        }
        
        search_mode = mode_map.get(mode, SearchMode.HYBRID)
        
        # Execute search
        best_move, search_info = self.hybrid_engine.search_position(
            self.board, time_limit=time_limit, depth=depth, mode=search_mode
        )
        
        return best_move, search_info
    
    def _search_classical(self, time_limit: float, depth: int) -> tuple:
        """Fallback to classical search."""
        from core.search import SearchEngine, SearchInfo
        
        classical_engine = SearchEngine(self.evaluator, self.move_generator)
        search_info = SearchInfo()
        search_info.max_depth = depth
        search_info.time_limit = time_limit
        
        best_move = classical_engine.search(self.board, search_info)
        
        return best_move, {
            'engine': 'classical',
            'depth': search_info.depth_reached,
            'nodes': search_info.nodes_searched,
            'time': search_info.time_used,
            'evaluation': search_info.best_score,
            'pv': search_info.principal_variation
        }
    
    def _send_search_info(self, search_info: Dict, elapsed_time: float):
        """Send search information to UCI."""
        engine_type = search_info.get('engine', 'unknown')
        
        if engine_type == 'classical':
            depth = search_info.get('depth', 0)
            nodes = search_info.get('nodes', 0)
            score = search_info.get('evaluation', 0)
            pv = search_info.get('pv', [])
            
            nps = int(nodes / max(elapsed_time, 0.001))
            time_ms = int(elapsed_time * 1000)
            
            pv_str = " ".join(str(move) for move in pv[:10])  # Limit PV length
            
            self._send(f"info depth {depth} nodes {nodes} time {time_ms} "
                      f"nps {nps} score cp {score} pv {pv_str}")
                      
        elif engine_type == 'mcts':
            simulations = search_info.get('simulations', 0)
            nps = search_info.get('nps', 0)
            pv = search_info.get('principal_variation', [])
            
            time_ms = int(elapsed_time * 1000)
            pv_str = " ".join(str(move) for move in pv[:10])
            
            self._send(f"info depth 0 nodes {simulations} time {time_ms} "
                      f"nps {int(nps)} pv {pv_str}")
        
        elif engine_type == 'hybrid':
            # Show information from the selected sub-engine
            selected_info = search_info.get('selected', {})
            self._send_search_info(selected_info, elapsed_time)
    
    # Enhanced UCI commands
    def _handle_analyze(self, parts: List[str]):
        """Handle position analysis command."""
        if not self.hybrid_engine:
            self._send("info string Analysis not available without neural network")
            return
        
        try:
            analysis = self.hybrid_engine.analyze_position(self.board)
            
            self._send(f"info string Position value: {analysis.get('position_value', 0):.3f}")
            self._send(f"info string Legal moves: {analysis.get('legal_move_count', 0)}")
            
            # Show best moves
            best_moves = analysis.get('best_moves', [])[:5]
            for i, move_info in enumerate(best_moves):
                move_str = move_info.get('move_str', '')
                prob = move_info.get('probability', 0)
                self._send(f"info string {i+1}. {move_str} (prob: {prob:.3f})")
                
        except Exception as e:
            self._send(f"info string Analysis error: {e}")
    
    def _handle_eval(self):
        """Handle position evaluation command."""
        try:
            # Classical evaluation
            classical_eval = self.evaluator.evaluate(self.board)
            self._send(f"info string Classical eval: {classical_eval}")
            
            # Neural network evaluation if available
            if self.neural_net:
                neural_eval = self.neural_net.evaluate_position(self.board)
                self._send(f"info string Neural eval: {neural_eval:.3f}")
                
        except Exception as e:
            self._send(f"info string Evaluation error: {e}")
    
    def _handle_perft(self, parts: List[str]):
        """Handle perft testing command."""
        if len(parts) < 2:
            self._send("info string Usage: perft <depth>")
            return
        
        try:
            depth = int(parts[1])
            if depth > 6:
                self._send("info string Depth limited to 6 for performance")
                depth = 6
            
            start_time = time.time()
            nodes = self._perft(self.board, depth)
            elapsed = time.time() - start_time
            
            self._send(f"info string Perft({depth}): {nodes} nodes in {elapsed:.3f}s")
            if elapsed > 0:
                self._send(f"info string Speed: {int(nodes/elapsed)} nodes/sec")
                
        except ValueError:
            self._send("info string Invalid depth")
        except Exception as e:
            self._send(f"info string Perft error: {e}")
    
    def _perft(self, board: ChessBoard, depth: int) -> int:
        """Perform perft test."""
        if depth == 0:
            return 1
        
        moves = self.move_generator.generate_legal_moves(board)
        total_nodes = 0
        
        for move in moves:
            board_copy = board.copy()
            board_copy.make_move(move)
            total_nodes += self._perft(board_copy, depth - 1)
        
        return total_nodes
    
    def _handle_bench(self, parts: List[str]):
        """Handle benchmark command."""
        self._send("info string Running benchmark...")
        
        # Simple benchmark - search starting position
        start_time = time.time()
        
        if self.hybrid_engine:
            best_move, search_info = self.hybrid_engine.search_position(
                self.board, time_limit=5.0
            )
            elapsed = time.time() - start_time
            
            nodes = search_info.get('nodes', search_info.get('simulations', 0))
            nps = int(nodes / max(elapsed, 0.001))
            
            self._send(f"info string Benchmark: {nodes} nodes in {elapsed:.2f}s")
            self._send(f"info string Speed: {nps} nodes/sec")
        else:
            self._send("info string Benchmark requires search engine")
      # Additional UCI command handlers
    def _handle_debug(self, parts: List[str]):
        """Handle 'debug' command."""
        if len(parts) > 1:
            self.options.debug_mode = parts[1].lower() == "on"
    
    def _handle_isready(self):
        """Handle 'isready' command."""
        self._send("readyok")
    
    def _handle_register(self):
        """Handle 'register' command."""
        self._send("registration ok")
    
    def _handle_ucinewgame(self):
        """Handle 'ucinewgame' command."""
        self.board = ChessBoard()
        self.analysis_cache.clear()
        self.search_history.clear()
    
    def _handle_position(self, parts: List[str]):
        """Handle 'position' command."""
        if len(parts) < 2:
            return
        
        if parts[1] == "startpos":
            self.board = ChessBoard()
            move_start = 2
            if len(parts) > 2 and parts[2] == "moves":
                move_start = 3
        elif parts[1] == "fen":
            # Find "moves" keyword or end of FEN
            fen_parts = []
            move_start = len(parts)
            
            for i in range(2, len(parts)):
                if parts[i] == "moves":
                    move_start = i + 1
                    break
                fen_parts.append(parts[i])
            
            if fen_parts:
                fen_string = " ".join(fen_parts)
                try:
                    self.board = ChessBoard(fen_string)
                except Exception as e:
                    if self.options.debug_mode:
                        self._send(f"info string Invalid FEN: {e}")
                    return
        
        # Apply moves
        for i in range(move_start, len(parts)):
            move_str = parts[i]
            try:
                move = self.board.parse_move(move_str)
                if move:
                    self.board.make_move(move)
                else:
                    if self.options.debug_mode:
                        self._send(f"info string Invalid move: {move_str}")
                    break
            except Exception as e:
                if self.options.debug_mode:
                    self._send(f"info string Error parsing move '{move_str}': {e}")
                break
    
    def _handle_stop(self):
        """Handle 'stop' command."""
        self.stop_search = True
        if self.search_thread and self.search_thread.is_alive():
            self.search_thread.join(timeout=1.0)
    
    def _handle_ponderhit(self):
        """Handle 'ponderhit' command."""
        # For now, just stop pondering
        pass
    
    def _handle_quit(self):
        """Handle 'quit' command."""
        self.stop_search = True
        if self.search_thread and self.search_thread.is_alive():
            self.search_thread.join(timeout=1.0)
        sys.exit(0)
    
    def _parse_go_params(self, parts: List[str]) -> Dict[str, Any]:
        """Parse 'go' command parameters."""
        params = {}
        i = 0
        
        while i < len(parts):
            param = parts[i].lower()
            
            if param in ["wtime", "btime", "winc", "binc", "movestogo"]:
                if i + 1 < len(parts):
                    try:
                        params[param] = int(parts[i + 1])
                        i += 2
                    except ValueError:
                        i += 1
                else:
                    i += 1
            elif param in ["depth", "nodes", "mate", "movetime"]:
                if i + 1 < len(parts):
                    try:
                        params[param] = int(parts[i + 1])
                        i += 2
                    except ValueError:
                        i += 1
                else:
                    i += 1
            elif param in ["infinite", "ponder"]:
                params[param] = True
                i += 1
            elif param == "searchmoves":
                # Parse list of moves to search
                i += 1
                searchmoves = []
                while i < len(parts) and not parts[i].startswith("-"):
                    searchmoves.append(parts[i])
                    i += 1
                params["searchmoves"] = searchmoves
            else:
                i += 1
        
        return params
    
    def _calculate_time_limit(self, params: Dict[str, Any]) -> float:
        """Calculate time limit for search."""
        if "movetime" in params:
            return params["movetime"] / 1000.0
        
        if "infinite" in params:
            return float('inf')
        
        # Use current player's time
        if self.board.current_player == Color.WHITE:
            time_left = params.get("wtime", 300000)  # Default 5 minutes
            increment = params.get("winc", 0)
        else:
            time_left = params.get("btime", 300000)
            increment = params.get("binc", 0)
        
        moves_to_go = params.get("movestogo", 40)
        
        # Simple time management: use 1/30th of remaining time plus increment
        base_time = time_left / 30000.0  # Convert to seconds
        time_limit = base_time + (increment / 1000.0)
        
        # Apply minimum and maximum limits
        time_limit = max(0.1, min(time_limit, time_left / 1000.0 * 0.8))
        
        return time_limit
    
    def _send(self, message: str):
        """Send message to UCI interface."""
        print(message, flush=True)


def main():
    """Main entry point for enhanced UCI engine."""
    print("Starting Enhanced Chess-AI UCI Engine...")
    engine = EnhancedUCIEngine()
    engine.run()


if __name__ == "__main__":
    main()