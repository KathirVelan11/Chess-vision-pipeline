"""
Utility functions for Chess Recognition System
==============================================

This module contains helper functions, model loading utilities, and
common operations used by the chess recognition system.

Author: GitHub Copilot
Date: October 8, 2025
"""

import cv2
import numpy as np
import torch
import os
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)

class ModelPathFinder:
    """Utility class to find model files automatically"""
    
    @staticmethod
    def find_piece_model(base_dir: str = ".") -> Optional[str]:
        """Find the best available piece classification model"""
        search_paths = [
            "piece-classification/chess_detection/*/weights/best.pt",
            "piece-classification/chess_detection/*/weights/last.pt",
            "piece-classification/yolov8n.pt",
            "piece-classification/yolo11n.pt"
        ]
        
        import glob
        for pattern in search_paths:
            full_pattern = os.path.join(base_dir, pattern)
            matches = glob.glob(full_pattern)
            if matches:
                # Sort by modification time, newest first
                matches.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                logger.info(f"Found piece model: {matches[0]}")
                return matches[0]
        
        logger.warning("No piece classification model found")
        return None
    
    @staticmethod
    def find_occupancy_model(base_dir: str = ".") -> Optional[str]:
        """Find the best available occupancy classification model"""
        search_paths = [
            "occupancy_classification/models/best_model.pth",
            "occupancy_classification/models/*.pth"
        ]
        
        import glob
        for pattern in search_paths:
            full_pattern = os.path.join(base_dir, pattern)
            matches = glob.glob(full_pattern)
            if matches:
                matches.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                logger.info(f"Found occupancy model: {matches[0]}")
                return matches[0]
        
        logger.warning("No occupancy classification model found")
        return None

class ImagePreprocessor:
    """Image preprocessing utilities"""
    
    @staticmethod
    def resize_image(image: np.ndarray, max_size: int = 1024) -> np.ndarray:
        """Resize image while maintaining aspect ratio"""
        h, w = image.shape[:2]
        if max(h, w) <= max_size:
            return image
        
        scale = max_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    @staticmethod
    def enhance_contrast(image: np.ndarray, alpha: float = 1.2, beta: int = 10) -> np.ndarray:
        """Enhance image contrast"""
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    @staticmethod
    def denoise_image(image: np.ndarray) -> np.ndarray:
        """Apply denoising to the image"""
        return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

class BoardValidator:
    """Validate detected board corners and squares"""
    
    @staticmethod
    def validate_corners(corners: List[Tuple[int, int]], 
                        image_shape: Tuple[int, int]) -> bool:
        """Validate that corners form a reasonable quadrilateral"""
        if len(corners) != 4:
            return False
        
        h, w = image_shape[:2]
        
        # Check if all corners are within image bounds
        for x, y in corners:
            if x < 0 or x >= w or y < 0 or y >= h:
                return False
        
        # Check if corners form a reasonable quadrilateral
        # (not too small, not degenerate)
        corners_array = np.array(corners)
        area = cv2.contourArea(corners_array)
        min_area = (min(w, h) * 0.1) ** 2  # At least 10% of smaller dimension
        
        return area > min_area
    
    @staticmethod
    def validate_board_squares(squares: List[List]) -> Dict[str, bool]:
        """Validate extracted chess squares"""
        validation = {
            'correct_dimensions': len(squares) == 8 and all(len(row) == 8 for row in squares),
            'valid_coordinates': True,
            'reasonable_occupancy': True
        }
        
        if not validation['correct_dimensions']:
            return validation
        
        # Check coordinates
        for row in squares:
            for square in row:
                if square.x1 >= square.x2 or square.y1 >= square.y2:
                    validation['valid_coordinates'] = False
                    break
        
        # Check occupancy (should have some pieces but not all squares)
        occupied_count = sum(1 for row in squares for square in row if square.is_occupied)
        total_squares = 64
        occupancy_rate = occupied_count / total_squares
        
        # Reasonable occupancy: between 10% and 90%
        if not (0.1 <= occupancy_rate <= 0.9):
            validation['reasonable_occupancy'] = False
        
        return validation

class FENValidator:
    """Validate and manipulate FEN strings"""
    
    @staticmethod
    def validate_fen(fen: str) -> Dict[str, bool]:
        """Validate FEN string format and content"""
        validation = {
            'format': False,
            'board': False,
            'pieces': False,
            'structure': False
        }
        
        try:
            parts = fen.split()
            if len(parts) != 6:
                return validation
            
            validation['format'] = True
            
            # Validate board part
            board_part = parts[0]
            ranks = board_part.split('/')
            
            if len(ranks) != 8:
                return validation
            
            valid_pieces = set('KQRBNPkqrbnp')
            total_pieces = 0
            
            for rank in ranks:
                rank_squares = 0
                for char in rank:
                    if char.isdigit():
                        rank_squares += int(char)
                    elif char in valid_pieces:
                        rank_squares += 1
                        total_pieces += 1
                    else:
                        return validation
                
                if rank_squares != 8:
                    return validation
            
            validation['board'] = True
            validation['pieces'] = 2 <= total_pieces <= 32  # Reasonable piece count
            validation['structure'] = True
            
        except Exception:
            pass
        
        return validation
    
    @staticmethod
    def fen_to_board_array(fen: str) -> Optional[np.ndarray]:
        """Convert FEN string to 8x8 board array"""
        try:
            board_part = fen.split()[0]
            ranks = board_part.split('/')
            
            board = np.empty((8, 8), dtype=object)
            
            for rank_idx, rank in enumerate(ranks):
                file_idx = 0
                for char in rank:
                    if char.isdigit():
                        for _ in range(int(char)):
                            board[rank_idx, file_idx] = None
                            file_idx += 1
                    else:
                        board[rank_idx, file_idx] = char
                        file_idx += 1
            
            return board
            
        except Exception as e:
            logger.error(f"Failed to parse FEN: {e}")
            return None

class PerformanceMonitor:
    """Monitor and log performance metrics"""
    
    def __init__(self):
        self.metrics = {}
    
    def start_timer(self, operation: str):
        """Start timing an operation"""
        import time
        self.metrics[operation] = {'start': time.time()}
    
    def end_timer(self, operation: str):
        """End timing an operation"""
        import time
        if operation in self.metrics:
            self.metrics[operation]['end'] = time.time()
            self.metrics[operation]['duration'] = (
                self.metrics[operation]['end'] - self.metrics[operation]['start']
            )
    
    def get_timing(self, operation: str) -> float:
        """Get timing for an operation"""
        return self.metrics.get(operation, {}).get('duration', 0.0)
    
    def get_all_timings(self) -> Dict[str, float]:
        """Get all timing information"""
        return {op: data.get('duration', 0.0) for op, data in self.metrics.items()}
    
    def log_performance(self):
        """Log performance metrics"""
        logger.info("Performance Metrics:")
        for operation, duration in self.get_all_timings().items():
            logger.info(f"  {operation}: {duration:.3f}s")

class ConfigManager:
    """Configuration management utilities"""
    
    @staticmethod
    def create_default_config(output_path: str = "config.json"):
        """Create a default configuration file"""
        default_config = {
            "piece_model_path": None,  # Will be auto-detected
            "occupancy_model_path": None,  # Will be auto-detected
            "square_size": 80,
            "piece_confidence": 0.3,
            "piece_iou": 0.5,
            "output_dir": "results",
            "image_preprocessing": {
                "max_size": 1024,
                "enhance_contrast": True,
                "denoise": False
            },
            "board_detection": {
                "edge_method": "scharr",
                "hough_threshold": 30,
                "min_line_length": 30,
                "max_line_gap": 5
            },
            "validation": {
                "validate_corners": True,
                "validate_squares": True,
                "validate_fen": True
            },
            "visualization": {
                "save_intermediate_steps": True,
                "show_confidence_scores": True,
                "square_border_color": [0, 255, 0],
                "piece_label_color": [255, 255, 255]
            }
        }
        
        # Auto-detect model paths
        piece_model = ModelPathFinder.find_piece_model()
        occupancy_model = ModelPathFinder.find_occupancy_model()
        
        if piece_model:
            default_config["piece_model_path"] = piece_model
        if occupancy_model:
            default_config["occupancy_model_path"] = occupancy_model
        
        # Save configuration
        with open(output_path, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        logger.info(f"Default configuration created: {output_path}")
        return default_config
    
    @staticmethod
    def load_config(config_path: str = None) -> Dict:
        """Load configuration with fallbacks"""
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                logger.info(f"Configuration loaded from: {config_path}")
                return config
            except Exception as e:
                logger.error(f"Failed to load config from {config_path}: {e}")
        
        # Create default config if none exists
        logger.info("Using default configuration")
        return ConfigManager.create_default_config()

class ResultsExporter:
    """Export recognition results in various formats"""
    
    @staticmethod
    def export_to_pgn(result: Dict, game_info: Dict = None) -> str:
        """Export result to PGN format"""
        if not result['success']:
            raise ValueError("Cannot export failed recognition result")
        
        # Default game info
        default_info = {
            'Event': 'Chess Position Recognition',
            'Site': 'Computer Vision System',
            'Date': '2025.10.08',
            'Round': '1',
            'White': 'Unknown',
            'Black': 'Unknown',
            'Result': '*'
        }
        
        if game_info:
            default_info.update(game_info)
        
        pgn_lines = []
        for key, value in default_info.items():
            pgn_lines.append(f'[{key} "{value}"]')
        
        pgn_lines.append('')
        pgn_lines.append(f'[FEN "{result["fen"]}"]')
        pgn_lines.append('[SetUp "1"]')
        pgn_lines.append('')
        pgn_lines.append('*')  # Game termination
        
        return '\n'.join(pgn_lines)
    
    @staticmethod
    def export_to_csv(results: List[Dict], output_path: str):
        """Export multiple results to CSV format"""
        import csv
        
        if not results:
            return
        
        with open(output_path, 'w', newline='') as csvfile:
            fieldnames = [
                'image_path', 'success', 'fen', 'total_squares', 
                'occupied_squares', 'identified_pieces', 'average_confidence',
                'recognition_rate', 'error'
            ]
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for result_data in results:
                image_path = result_data.get('image_path', '')
                result = result_data.get('result', {})
                
                row = {
                    'image_path': image_path,
                    'success': result.get('success', False),
                    'fen': result.get('fen', ''),
                    'error': result.get('error', '')
                }
                
                if 'statistics' in result:
                    stats = result['statistics']
                    row.update({
                        'total_squares': stats.get('total_squares', 0),
                        'occupied_squares': stats.get('occupied_squares', 0),
                        'identified_pieces': stats.get('identified_pieces', 0),
                        'average_confidence': stats.get('average_confidence', 0),
                        'recognition_rate': stats.get('recognition_rate', 0)
                    })
                
                writer.writerow(row)
        
        logger.info(f"Results exported to CSV: {output_path}")

# Common utility functions
def chess_position_to_coordinates(position: str) -> Tuple[int, int]:
    """Convert chess position (e.g., 'e4') to row, col coordinates"""
    if len(position) != 2:
        raise ValueError("Position must be 2 characters (e.g., 'e4')")
    
    file_char, rank_char = position.lower()
    
    col = ord(file_char) - ord('a')  # 0-7
    row = 8 - int(rank_char)  # 0-7 (0 = rank 8, 7 = rank 1)
    
    if not (0 <= col <= 7) or not (0 <= row <= 7):
        raise ValueError(f"Invalid position: {position}")
    
    return row, col

def coordinates_to_chess_position(row: int, col: int) -> str:
    """Convert row, col coordinates to chess position (e.g., 'e4')"""
    if not (0 <= row <= 7) or not (0 <= col <= 7):
        raise ValueError(f"Invalid coordinates: ({row}, {col})")
    
    file_char = chr(ord('a') + col)
    rank_char = str(8 - row)
    
    return file_char + rank_char

def setup_logging(level: str = 'INFO', log_file: str = None):
    """Setup logging configuration"""
    logging_level = getattr(logging, level.upper(), logging.INFO)
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=logging_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

# Example usage and testing
if __name__ == "__main__":
    # Test utilities
    setup_logging('DEBUG')
    
    # Test model finding
    piece_model = ModelPathFinder.find_piece_model()
    occupancy_model = ModelPathFinder.find_occupancy_model()
    
    print(f"Found piece model: {piece_model}")
    print(f"Found occupancy model: {occupancy_model}")
    
    # Test configuration creation
    config = ConfigManager.create_default_config("test_config.json")
    print(f"Created config with {len(config)} settings")
    
    # Test coordinate conversion
    try:
        row, col = chess_position_to_coordinates('e4')
        pos = coordinates_to_chess_position(row, col)
        print(f"e4 -> ({row}, {col}) -> {pos}")
    except ValueError as e:
        print(f"Error: {e}")
    
    # Test FEN validation
    test_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    validation = FENValidator.validate_fen(test_fen)
    print(f"FEN validation: {validation}")
