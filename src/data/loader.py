"""
Chess Game Data Loader and Analyzer
===================================

This module handles loading and initial analysis of chess game data.
Part of Phase 1: Foundational Engine & Initial Data Processing.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChessDataLoader:
    """Handles loading and analysis of chess game data."""
    
    def __init__(self, data_path: str):
        """Initialize with path to chess data CSV."""
        self.data_path = Path(data_path)
        self.data: Optional[pd.DataFrame] = None
        
    def load_data(self) -> pd.DataFrame:
        """Load chess game data from CSV file."""
        try:
            logger.info(f"Loading chess data from {self.data_path}")
            self.data = pd.read_csv(self.data_path)
            logger.info(f"Loaded {len(self.data)} games")
            return self.data
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise
    
    def analyze_data_structure(self) -> Dict:
        """Analyze the structure and content of the loaded data."""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        analysis = {
            'total_games': len(self.data),
            'columns': list(self.data.columns),
            'data_types': dict(self.data.dtypes),
            'missing_values': dict(self.data.isnull().sum()),
            'unique_openings': self.data['opening_name'].nunique() if 'opening_name' in self.data.columns else 0,
            'game_outcomes': dict(self.data['victory_status'].value_counts()) if 'victory_status' in self.data.columns else {},
            'winner_distribution': dict(self.data['winner'].value_counts()) if 'winner' in self.data.columns else {},
        }
        
        return analysis
    
    def analyze_game_characteristics(self) -> Dict:
        """Analyze game characteristics like length, openings, outcomes."""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Calculate game length in half-moves (ply)
        if 'turns' in self.data.columns:
            avg_game_length = self.data['turns'].mean()
            median_game_length = self.data['turns'].median()
            max_game_length = self.data['turns'].max()
            min_game_length = self.data['turns'].min()
        else:
            avg_game_length = median_game_length = max_game_length = min_game_length = None
        
        # Opening analysis
        opening_stats = {}
        if 'opening_eco' in self.data.columns:
            opening_stats = {
                'most_common_eco': dict(self.data['opening_eco'].value_counts().head(10)),
                'eco_categories': dict(self.data['opening_eco'].str[0].value_counts())
            }
        
        return {
            'game_length_stats': {
                'average': avg_game_length,
                'median': median_game_length,
                'max': max_game_length,
                'min': min_game_length
            },
            'opening_analysis': opening_stats
        }
    
    def extract_training_positions(self) -> List[Dict]:
        """Extract positions and outcomes for training data preparation."""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        training_positions = []
        
        for idx, game in self.data.iterrows():
            if 'moves' not in game or pd.isna(game['moves']):
                continue
                
            try:
                moves = game['moves'].split()
                winner = game.get('winner', 'draw')
                victory_status = game.get('victory_status', 'unknown')
                
                # Convert winner to numeric score for training
                if winner == 'white':
                    outcome = 1.0
                elif winner == 'black':
                    outcome = -1.0
                else:
                    outcome = 0.0
                
                training_positions.append({
                    'game_id': idx,
                    'moves': moves,
                    'outcome': outcome,
                    'victory_status': victory_status,
                    'opening_eco': game.get('opening_eco', ''),
                    'opening_name': game.get('opening_name', ''),
                    'game_length': len(moves)
                })
                
            except Exception as e:
                logger.warning(f"Error processing game {idx}: {e}")
                continue
        
        logger.info(f"Extracted {len(training_positions)} training positions")
        return training_positions
    
    def generate_data_report(self, output_path: Optional[str] = None) -> str:
        """Generate a comprehensive data analysis report."""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        structure_analysis = self.analyze_data_structure()
        game_analysis = self.analyze_game_characteristics()
        
        report = f"""
Chess AI Training Data Analysis Report
====================================

Dataset Overview:
- Total Games: {structure_analysis['total_games']:,}
- Columns: {', '.join(structure_analysis['columns'])}
- Unique Openings: {structure_analysis['unique_openings']:,}

Game Length Statistics:
- Average: {game_analysis['game_length_stats']['average']:.1f} moves
- Median: {game_analysis['game_length_stats']['median']:.1f} moves
- Range: {game_analysis['game_length_stats']['min']} - {game_analysis['game_length_stats']['max']} moves

Game Outcomes:
"""
        
        for outcome, count in structure_analysis['game_outcomes'].items():
            percentage = (count / structure_analysis['total_games']) * 100
            report += f"- {outcome}: {count:,} ({percentage:.1f}%)\n"
        
        report += "\nWinner Distribution:\n"
        for winner, count in structure_analysis['winner_distribution'].items():
            percentage = (count / structure_analysis['total_games']) * 100
            report += f"- {winner}: {count:,} ({percentage:.1f}%)\n"
        
        report += "\nTop Opening ECO Codes:\n"
        if game_analysis['opening_analysis']:
            for eco, count in list(game_analysis['opening_analysis']['most_common_eco'].items())[:5]:
                percentage = (count / structure_analysis['total_games']) * 100
                report += f"- {eco}: {count:,} ({percentage:.1f}%)\n"
        
        report += f"""
Data Quality Assessment:
- Missing values found in: {[col for col, missing in structure_analysis['missing_values'].items() if missing > 0]}
- Data appears suitable for initial training with {structure_analysis['total_games']:,} complete games
- Recommended next steps: Expand dataset to millions of games for production-level training

Recommended Data Augmentation:
1. Acquire larger corpus from chess databases (FICS, lichess, chess.com)
2. Add high-depth engine evaluations for each position
3. Extract tactical puzzle datasets
4. Include endgame tablebase positions
"""
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to {output_path}")
        
        return report


def main():
    """Main function to analyze chess data."""
    # Initialize data loader
    data_path = "trainning-data/Chess-game-data.csv"  # Note: keeping original spelling
    loader = ChessDataLoader(data_path)
    
    try:
        # Load and analyze data
        data = loader.load_data()
        print("✅ Data loaded successfully")
        
        # Generate analysis report
        report = loader.generate_data_report("analysis_report.txt")
        print(report)
        
        # Extract training positions for future use
        positions = loader.extract_training_positions()
        print(f"✅ Extracted {len(positions)} training positions")
        
        # Save processed positions for neural network training
        import json
        with open("models/initial_training_positions.json", "w") as f:
            json.dump(positions, f, indent=2)
        print("✅ Training positions saved to models/initial_training_positions.json")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
