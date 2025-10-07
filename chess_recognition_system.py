"""
Chess State Recognition System - Complete Integration
===================================================

This module integrates board localization, occupancy classification, and piece 
classification to determine the complete chess state from a single image and 
output it in FEN (Forsyth-Edwards Notation) format.

Based on the paper: "Determining Chess Game State From an Image" by Wölflein & Arandjelović

Author: GitHub Copilot
Date: October 8, 2025
"""

import cv2
import numpy as np
import torch
import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import json
import argparse
from ultralytics import YOLO
import torch.nn as nn
import torchvision.models as models
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ChessSquare:
    """Represents a chess square with its position and content"""
    row: int  # 0-7 (0 = rank 8, 7 = rank 1)
    col: int  # 0-7 (0 = file a, 7 = file h)
    x1: int   # Bounding box coordinates
    y1: int
    x2: int
    y2: int
    is_occupied: bool = False
    piece: Optional[str] = None  # e.g., 'white-king', 'black-pawn'
    confidence: float = 0.0

@dataclass
class ChessBoard:
    """Represents the complete chess board state"""
    squares: List[List[ChessSquare]]  # 8x8 grid
    corner_points: List[Tuple[int, int]]  # Four corner points of the board
    
    def get_square(self, row: int, col: int) -> ChessSquare:
        """Get square at given position"""
        return self.squares[row][col]
    
    def to_fen(self) -> str:
        """Convert board state to FEN notation"""
        fen_rows = []
        
        for row in range(8):  # From rank 8 to 1
            fen_row = ""
            empty_count = 0
            
            for col in range(8):  # From file a to h
                square = self.squares[row][col]
                
                if not square.is_occupied or square.piece is None:
                    empty_count += 1
                else:
                    if empty_count > 0:
                        fen_row += str(empty_count)
                        empty_count = 0
                    
                    # Convert piece name to FEN notation
                    piece_char = self._piece_to_fen_char(square.piece)
                    fen_row += piece_char
            
            if empty_count > 0:
                fen_row += str(empty_count)
            
            fen_rows.append(fen_row)
        
        # Join rows with '/' and add default game state info
        board_fen = "/".join(fen_rows)
        # Add default values for: active color, castling, en passant, halfmove, fullmove
        return f"{board_fen} w - - 0 1"
    
    def _piece_to_fen_char(self, piece_name: str) -> str:
        """Convert piece name to single character FEN notation"""
        piece_map = {
            'white-king': 'K', 'white-queen': 'Q', 'white-rook': 'R',
            'white-bishop': 'B', 'white-knight': 'N', 'white-pawn': 'P',
            'black-king': 'k', 'black-queen': 'q', 'black-rook': 'r',
            'black-bishop': 'b', 'black-knight': 'n', 'black-pawn': 'p',
            # Handle the generic 'bishop' class from your data.yaml
            'bishop': 'B'  # Default to white, but this should be improved
        }
        return piece_map.get(piece_name, '?')

class ChessRecognitionSystem:
    """Complete chess state recognition system"""
    
    def __init__(self, config: Dict):
        """Initialize the recognition system with configuration"""
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load models
        self._load_piece_classifier()
        self._load_occupancy_classifier()
        
        logger.info(f"Chess Recognition System initialized on {self.device}")
    
    def _load_piece_classifier(self):
        """Load the YOLO piece classification model"""
        weights_path = self.config.get('piece_model_path', 'piece-classification/yolov8n.pt')
        
        try:
            self.piece_model = YOLO(weights_path)
            logger.info(f"Piece classifier loaded from {weights_path}")
        except Exception as e:
            logger.error(f"Failed to load piece classifier: {e}")
            raise
    
    def _load_occupancy_classifier(self):
        """Load the occupancy classification model"""
        model_path = self.config.get('occupancy_model_path')
        
        if not model_path or not os.path.exists(model_path):
            logger.warning("Occupancy classifier not found, will use piece detection only")
            self.occupancy_model = None
            return
        
        try:
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Create model architecture
            self.occupancy_model = models.resnet18(weights=None)
            self.occupancy_model.fc = nn.Linear(self.occupancy_model.fc.in_features, 2)
            
            # Load trained weights
            self.occupancy_model.load_state_dict(checkpoint['model_state_dict'])
            self.occupancy_model.to(self.device)
            self.occupancy_model.eval()
            
            # Define transforms
            self.occupancy_transform = A.Compose([
                A.Resize(224, 224),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
            
            logger.info(f"Occupancy classifier loaded from {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load occupancy classifier: {e}")
            self.occupancy_model = None
    
    def detect_board_corners(self, image: np.ndarray) -> List[Tuple[int, int]]:
        """
        Detect chess board corners using the board localization module
        
        Returns:
            List of 4 corner points in order: top-left, top-right, bottom-left, bottom-right
        """
        # Import board detection function
        import sys
        sys.path.append('board-localisation')
        
        try:
            from board_localization.board_detection import detect_edges, enhanced_probabilistic_hough, direct_line_intersection
            
            # Edge detection
            edges = detect_edges(image, method='scharr')
            
            # Line detection
            lines = enhanced_probabilistic_hough(edges)
            
            # Find intersections
            intersections = direct_line_intersection(lines, image.shape)
            
            if len(intersections) < 4:
                raise ValueError("Could not find enough intersections for board corners")
            
            # Convert Point objects to tuples and find the 4 corners
            points = [(p.x, p.y) for p in intersections]
            corners = self._find_board_corners(points, image.shape)
            
            return corners
            
        except Exception as e:
            logger.error(f"Board corner detection failed: {e}")
            # Return default corners as fallback
            h, w = image.shape[:2]
            margin = min(w, h) // 10
            return [
                (margin, margin),  # top-left
                (w - margin, margin),  # top-right
                (margin, h - margin),  # bottom-left
                (w - margin, h - margin)  # bottom-right
            ]
    
    def _find_board_corners(self, points: List[Tuple[int, int]], img_shape: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Find the 4 corners of the chess board from intersection points"""
        if len(points) < 4:
            raise ValueError("Need at least 4 points to find corners")
        
        h, w = img_shape[:2]
        points = np.array(points)
        
        # Find corners using geometric properties
        # Top-left: minimum sum of coordinates
        top_left = points[np.argmin(points.sum(axis=1))]
        
        # Bottom-right: maximum sum of coordinates  
        bottom_right = points[np.argmax(points.sum(axis=1))]
        
        # Top-right: minimum difference (x - y)
        top_right = points[np.argmin(points[:, 0] - points[:, 1])]
        
        # Bottom-left: maximum difference (x - y)
        bottom_left = points[np.argmax(points[:, 0] - points[:, 1])]
        
        return [
            tuple(top_left.astype(int)),
            tuple(top_right.astype(int)),
            tuple(bottom_left.astype(int)),
            tuple(bottom_right.astype(int))
        ]
    
    def extract_chess_squares(self, image: np.ndarray, corners: List[Tuple[int, int]]) -> ChessBoard:
        """
        Extract individual chess squares from the board using perspective transformation
        
        Args:
            image: Input image
            corners: Four corner points of the board
            
        Returns:
            ChessBoard object with all squares extracted
        """
        # Define target square size
        square_size = self.config.get('square_size', 64)
        board_size = square_size * 8
        
        # Source points (corners)
        src_points = np.array(corners, dtype=np.float32)
        
        # Destination points (perfect square)
        dst_points = np.array([
            [0, 0],  # top-left
            [board_size, 0],  # top-right
            [0, board_size],  # bottom-left
            [board_size, board_size]  # bottom-right
        ], dtype=np.float32)
        
        # Calculate perspective transformation matrix
        transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # Apply perspective transformation
        warped_board = cv2.warpPerspective(image, transform_matrix, (board_size, board_size))
        
        # Extract individual squares
        squares = []
        for row in range(8):
            square_row = []
            for col in range(8):
                # Calculate square coordinates
                x1 = col * square_size
                y1 = row * square_size
                x2 = x1 + square_size
                y2 = y1 + square_size
                
                # Create square object
                square = ChessSquare(
                    row=row, col=col,
                    x1=x1, y1=y1, x2=x2, y2=y2
                )
                square_row.append(square)
            
            squares.append(square_row)
        
        # Create chess board object
        chess_board = ChessBoard(squares=squares, corner_points=corners)
        
        # Store the warped board image for later use
        self.warped_board = warped_board
        
        return chess_board
    
    def classify_occupancy(self, chess_board: ChessBoard) -> ChessBoard:
        """
        Classify each square as occupied or empty
        
        Args:
            chess_board: ChessBoard object with extracted squares
            
        Returns:
            Updated ChessBoard with occupancy information
        """
        if self.occupancy_model is None:
            logger.warning("Occupancy model not available, skipping occupancy classification")
            return chess_board
        
        for row in range(8):
            for col in range(8):
                square = chess_board.squares[row][col]
                
                # Extract square image
                square_img = self.warped_board[square.y1:square.y2, square.x1:square.x2]
                
                # Preprocess for occupancy model
                transformed = self.occupancy_transform(image=square_img)
                tensor_img = transformed['image'].unsqueeze(0).to(self.device)
                
                # Predict occupancy
                with torch.no_grad():
                    outputs = self.occupancy_model(tensor_img)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    confidence, predicted_class = torch.max(probabilities, 1)
                    
                    square.is_occupied = bool(predicted_class.item())
                    square.confidence = confidence.item()
        
        return chess_board
    
    def classify_pieces(self, chess_board: ChessBoard) -> ChessBoard:
        """
        Classify pieces in occupied squares
        
        Args:
            chess_board: ChessBoard object with occupancy information
            
        Returns:
            Updated ChessBoard with piece classifications
        """
        # Run YOLO on the warped board
        results = self.piece_model.predict(
            self.warped_board,
            conf=self.config.get('piece_confidence', 0.25),
            iou=self.config.get('piece_iou', 0.45),
            verbose=False
        )
        
        # Process detections
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    piece_name = result.names[cls_id]
                    
                    # Find which square this detection belongs to
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    
                    square_row = center_y // (self.warped_board.shape[0] // 8)
                    square_col = center_x // (self.warped_board.shape[1] // 8)
                    
                    # Ensure coordinates are within bounds
                    if 0 <= square_row < 8 and 0 <= square_col < 8:
                        square = chess_board.squares[square_row][square_col]
                        
                        # Update square with piece information (keep highest confidence)
                        if square.piece is None or conf > square.confidence:
                            square.piece = piece_name
                            square.confidence = conf
                            square.is_occupied = True
        
        return chess_board
    
    def recognize_chess_state(self, image_path: str) -> Dict:
        """
        Complete chess state recognition pipeline
        
        Args:
            image_path: Path to the chess board image
            
        Returns:
            Dictionary containing the complete chess state and FEN
        """
        logger.info(f"Processing image: {image_path}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        try:
            # Step 1: Detect board corners
            logger.info("Step 1: Detecting board corners...")
            corners = self.detect_board_corners(image)
            
            # Step 2: Extract chess squares
            logger.info("Step 2: Extracting chess squares...")
            chess_board = self.extract_chess_squares(image, corners)
            
            # Step 3: Classify occupancy (if model available)
            logger.info("Step 3: Classifying square occupancy...")
            chess_board = self.classify_occupancy(chess_board)
            
            # Step 4: Classify pieces
            logger.info("Step 4: Classifying pieces...")
            chess_board = self.classify_pieces(chess_board)
            
            # Step 5: Generate FEN
            logger.info("Step 5: Generating FEN notation...")
            fen = chess_board.to_fen()
            
            # Compile results
            result = {
                'success': True,
                'fen': fen,
                'board_corners': corners,
                'squares': [],
                'statistics': self._calculate_statistics(chess_board)
            }
            
            # Add square details
            for row in range(8):
                for col in range(8):
                    square = chess_board.squares[row][col]
                    result['squares'].append({
                        'position': f"{chr(ord('a') + col)}{8 - row}",  # e.g., "e4"
                        'row': row,
                        'col': col,
                        'is_occupied': square.is_occupied,
                        'piece': square.piece,
                        'confidence': square.confidence
                    })
            
            logger.info(f"Recognition completed successfully. FEN: {fen}")
            return result
            
        except Exception as e:
            logger.error(f"Recognition failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'fen': None
            }
    
    def _calculate_statistics(self, chess_board: ChessBoard) -> Dict:
        """Calculate statistics about the recognized board state"""
        total_squares = 64
        occupied_squares = sum(1 for row in chess_board.squares for square in row if square.is_occupied)
        identified_pieces = sum(1 for row in chess_board.squares for square in row if square.piece is not None)
        
        avg_confidence = 0
        confident_squares = 0
        for row in chess_board.squares:
            for square in row:
                if square.confidence > 0:
                    avg_confidence += square.confidence
                    confident_squares += 1
        
        avg_confidence = avg_confidence / confident_squares if confident_squares > 0 else 0
        
        return {
            'total_squares': total_squares,
            'occupied_squares': occupied_squares,
            'identified_pieces': identified_pieces,
            'average_confidence': avg_confidence,
            'recognition_rate': identified_pieces / total_squares
        }

def load_config(config_path: str = None) -> Dict:
    """Load configuration from file or return defaults"""
    default_config = {
        'piece_model_path': 'piece-classification/chess_detection/chess_model/weights/best.pt',
        'occupancy_model_path': 'occupancy_classification/models/best_model.pth',
        'square_size': 64,
        'piece_confidence': 0.25,
        'piece_iou': 0.45,
        'output_dir': 'results'
    }
    
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            user_config = json.load(f)
        default_config.update(user_config)
    
    return default_config

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='Chess State Recognition System')
    parser.add_argument('image_path', type=str, help='Path to chess board image')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--output', type=str, help='Output file for results (JSON)')
    parser.add_argument('--save-visualization', action='store_true', 
                       help='Save visualization of the recognition process')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize recognition system
    try:
        system = ChessRecognitionSystem(config)
        
        # Recognize chess state
        result = system.recognize_chess_state(args.image_path)
        
        if result['success']:
            print(f"Recognition successful!")
            print(f"FEN: {result['fen']}")
            print(f"Statistics: {result['statistics']}")
            
            # Save results if requested
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"Results saved to: {args.output}")
                
        else:
            print(f"Recognition failed: {result['error']}")
            
    except Exception as e:
        logger.error(f"System initialization failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
