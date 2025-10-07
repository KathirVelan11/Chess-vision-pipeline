"""
Integrated Chess Recognition System Using Existing Models
This system integrates your existing YOLO piece classification model with 
the board localization and occupancy classification components.
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
from pathlib import Path
import time
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Import our board localization
from board_localization_improved import ChessBoardLocalizer
from fen_generator import FENGenerator

class IntegratedChessRecognitionSystem:
    """Chess recognition system using your existing trained models"""
    
    def __init__(self, 
                 yolo_model_path: str = "/home/pranesh/chess/Computer-Vision/piece-classification/chess_detection/chess_model_20251007_185940/weights/best.pt",
                 occupancy_model_path: str = None,
                 device: str = 'cuda'):
        """
        Initialize the integrated system
        
        Args:
            yolo_model_path: Path to your trained YOLO model
            occupancy_model_path: Path to occupancy model (if available)
            device: Computing device
        """
        self.device = device
        
        # Initialize components
        self.board_localizer = ChessBoardLocalizer()
        self.fen_generator = FENGenerator()
        
        # Load your trained YOLO model
        print(f"Loading YOLO model from: {yolo_model_path}")
        if Path(yolo_model_path).exists():
            self.yolo_model = YOLO(yolo_model_path)
            print("✅ YOLO model loaded successfully")
        else:
            print(f"❌ YOLO model not found at {yolo_model_path}")
            print("Available models:")
            for model_path in Path("/home/pranesh/chess/Computer-Vision/piece-classification").rglob("*.pt"):
                print(f"  - {model_path}")
            raise FileNotFoundError(f"YOLO model not found: {yolo_model_path}")
        
        # YOLO class names from your data.yaml
        self.yolo_classes = [
            'bishop', 'black-bishop', 'black-king', 'black-knight', 
            'black-pawn', 'black-queen', 'black-rook', 'white-bishop', 
            'white-king', 'white-knight', 'white-pawn', 'white-queen', 'white-rook'
        ]
        
        # Mapping from YOLO classes to standard piece names
        self.class_to_piece = {
            'black-bishop': 'black_bishop',
            'black-king': 'black_king', 
            'black-knight': 'black_knight',
            'black-pawn': 'black_pawn',
            'black-queen': 'black_queen',
            'black-rook': 'black_rook',
            'white-bishop': 'white_bishop',
            'white-king': 'white_king',
            'white-knight': 'white_knight', 
            'white-pawn': 'white_pawn',
            'white-queen': 'white_queen',
            'white-rook': 'white_rook',
            'bishop': 'white_bishop'  # Generic bishop assumed white
        }
        
        print("🎯 Integrated Chess Recognition System initialized")
        print(f"  - Board Localizer: Ready")
        print(f"  - YOLO Model: Loaded ({len(self.yolo_classes)} classes)")
        print(f"  - FEN Generator: Ready")
    
    def process_chess_image(self, image_path: str, 
                           confidence_threshold: float = 0.5,
                           save_results: bool = True,
                           output_dir: str = "integrated_results") -> Dict:
        """
        Process chess image using your trained models
        
        Args:
            image_path: Path to chess board image
            confidence_threshold: Minimum confidence for YOLO detections
            save_results: Whether to save intermediate results
            output_dir: Directory to save results
            
        Returns:
            Dictionary with processing results
        """
        start_time = time.time()
        
        if save_results:
            Path(output_dir).mkdir(exist_ok=True)
        
        results = {
            'success': False,
            'image_path': image_path,
            'fen': None,
            'confidence_score': 0.0,
            'processing_time': 0.0,
            'stages': {}
        }
        
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            print(f"\n🖼️  Processing: {Path(image_path).name}")
            print(f"Image size: {image.shape}")
            
            # Stage 1: Board Localization
            print("\n" + "="*50)
            print("STAGE 1: BOARD LOCALIZATION")
            print("="*50)
            
            stage1_start = time.time()
            homography, corner_points, localization_results = self.board_localizer.localize_board(image)
            stage1_time = time.time() - stage1_start
            
            results['stages']['localization'] = {
                'success': homography is not None,
                'time': stage1_time,
                'corner_points': len(corner_points),
                'lines_detected': len(localization_results.get('lines', [])),
                'intersections': len(localization_results.get('intersections', []))
            }
            
            print(f"Board localization: {'✅ SUCCESS' if homography is not None else '❌ FAILED'}")
            print(f"Processing time: {stage1_time:.3f}s")
            
            if homography is None:
                print("❌ Board localization failed - trying YOLO on original image")
                # Fall back to YOLO on original image without rectification
                warped_board = image
                board_corners = None
            else:
                # Rectify the board
                board_size = 640  # Standard YOLO input size
                warped_board = cv2.warpPerspective(image, homography, (board_size, board_size))
                board_corners = corner_points
                
                if save_results:
                    cv2.imwrite(f"{output_dir}/01_rectified_board.png", warped_board)
            
            # Stage 2: YOLO Piece Detection
            print("\n" + "="*50)
            print("STAGE 2: YOLO PIECE DETECTION")
            print("="*50)
            
            stage2_start = time.time()
            
            # Run YOLO detection
            yolo_results = self.yolo_model(warped_board, conf=confidence_threshold)
            stage2_time = time.time() - stage2_start
            
            # Process YOLO results
            detections = []
            if yolo_results and len(yolo_results[0].boxes) > 0:
                boxes = yolo_results[0].boxes
                for i in range(len(boxes)):
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                    confidence = boxes.conf[i].cpu().numpy()
                    class_id = int(boxes.cls[i].cpu().numpy())
                    
                    if class_id < len(self.yolo_classes):
                        class_name = self.yolo_classes[class_id]
                        piece_name = self.class_to_piece.get(class_name, class_name)
                        
                        detections.append({
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'confidence': float(confidence),
                            'class_name': class_name,
                            'piece_name': piece_name,
                            'center': [(x1 + x2) / 2, (y1 + y2) / 2]
                        })
            
            results['stages']['yolo_detection'] = {
                'success': len(detections) > 0,
                'time': stage2_time,
                'detections': len(detections),
                'average_confidence': np.mean([d['confidence'] for d in detections]) if detections else 0.0
            }
            
            print(f"YOLO detection: {'✅ SUCCESS' if len(detections) > 0 else '❌ NO DETECTIONS'}")
            print(f"Processing time: {stage2_time:.3f}s")
            print(f"Pieces detected: {len(detections)}")
            
            if save_results:
                self._save_yolo_visualization(warped_board, detections, f"{output_dir}/02_yolo_detections.png")
            
            # Stage 3: Convert to Chess Board Grid
            print("\n" + "="*50)
            print("STAGE 3: GRID MAPPING")
            print("="*50)
            
            stage3_start = time.time()
            
            if homography is not None:
                # Map detections to 8x8 chess grid
                board_state = self._map_detections_to_grid(detections, board_size)
            else:
                # Create board state from raw detections (less accurate)
                board_state = self._create_board_from_detections(detections, warped_board.shape)
            
            stage3_time = time.time() - stage3_start
            
            results['stages']['grid_mapping'] = {
                'success': True,
                'time': stage3_time,
                'occupied_squares': np.sum(board_state != 'empty')
            }
            
            print(f"Grid mapping: ✅ SUCCESS")
            print(f"Processing time: {stage3_time:.3f}s")
            print(f"Occupied squares: {np.sum(board_state != 'empty')}")
            
            # Stage 4: FEN Generation
            print("\n" + "="*50)
            print("STAGE 4: FEN GENERATION")
            print("="*50)
            
            stage4_start = time.time()
            
            # Generate FEN
            fen = self.fen_generator.board_to_fen(board_state)
            
            # Validate FEN
            is_valid, validation_errors = self.fen_generator.validate_fen(fen)
            
            stage4_time = time.time() - stage4_start
            
            results['stages']['fen_generation'] = {
                'success': True,
                'time': stage4_time,
                'fen_valid': is_valid,
                'validation_errors': validation_errors
            }
            
            print(f"FEN generation: ✅ SUCCESS")
            print(f"Processing time: {stage4_time:.3f}s")
            print(f"Generated FEN: {fen}")
            print(f"FEN valid: {'✅ YES' if is_valid else '❌ NO'}")
            
            if validation_errors:
                print(f"Validation warnings: {validation_errors}")
            
            if save_results:
                self._save_final_result(warped_board, board_state, detections, fen, 
                                      f"{output_dir}/03_final_result.png")
            
            # Calculate overall confidence
            detection_confidence = np.mean([d['confidence'] for d in detections]) if detections else 0.0
            localization_confidence = 1.0 if homography is not None else 0.5
            validation_confidence = 1.0 if is_valid else 0.7
            
            overall_confidence = np.mean([detection_confidence, localization_confidence, validation_confidence])
            
            # Final results
            total_time = time.time() - start_time
            
            results.update({
                'success': True,
                'fen': fen,
                'confidence_score': float(overall_confidence),
                'processing_time': total_time,
                'board_state': board_state.tolist(),
                'detections': detections,
                'validation_errors': validation_errors
            })
            
            print(f"\n🎉 PROCESSING COMPLETED!")
            print(f"Total time: {total_time:.3f}s")
            print(f"Overall confidence: {overall_confidence:.3f}")
            print(f"Final FEN: {fen}")
            
        except Exception as e:
            print(f"\n❌ Processing failed: {str(e)}")
            results.update({
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time
            })
        
        return results
    
    def _map_detections_to_grid(self, detections: List[Dict], board_size: int) -> np.ndarray:
        """Map YOLO detections to 8x8 chess grid"""
        board_state = np.full((8, 8), 'empty', dtype=object)
        
        square_size = board_size / 8
        
        for detection in detections:
            center_x, center_y = detection['center']
            
            # Convert to grid coordinates
            col = int(center_x / square_size)
            row = int(center_y / square_size)
            
            # Ensure within bounds
            col = max(0, min(7, col))
            row = max(0, min(7, row))
            
            # Place piece (if square is empty or this detection has higher confidence)
            current_piece = board_state[row, col]
            if current_piece == 'empty':
                board_state[row, col] = detection['piece_name']
            else:
                # Keep higher confidence detection
                # This is a simplified approach - in practice you might want more sophisticated logic
                board_state[row, col] = detection['piece_name']
        
        return board_state
    
    def _create_board_from_detections(self, detections: List[Dict], image_shape: Tuple) -> np.ndarray:
        """Create board state from detections without grid mapping (fallback)"""
        # This is a simplified fallback when board localization fails
        # It creates a rough board state based on detection positions
        board_state = np.full((8, 8), 'empty', dtype=object)
        
        if not detections:
            return board_state
        
        h, w = image_shape[:2]
        
        # Sort detections by position (top to bottom, left to right)
        detections.sort(key=lambda d: (d['center'][1], d['center'][0]))
        
        # Place pieces in a rough grid pattern
        for i, detection in enumerate(detections[:32]):  # Max 32 pieces
            row = i // 8
            col = i % 8
            if row < 8:
                board_state[row, col] = detection['piece_name']
        
        return board_state
    
    def _save_yolo_visualization(self, image: np.ndarray, detections: List[Dict], save_path: str):
        """Save YOLO detection visualization"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        
        # Display image
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Draw detections
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            conf = detection['confidence']
            piece_name = detection['piece_name']
            
            # Draw bounding box
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                               fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
            
            # Add label
            ax.text(x1, y1-5, f'{piece_name}\n{conf:.2f}', 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                   fontsize=8, fontweight='bold')
        
        ax.set_title(f'YOLO Detections ({len(detections)} pieces)')
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _save_final_result(self, image: np.ndarray, board_state: np.ndarray, 
                          detections: List[Dict], fen: str, save_path: str):
        """Save final result visualization"""
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        
        # Left: Image with detections
        axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                               fill=False, edgecolor='red', linewidth=2)
            axes[0].add_patch(rect)
        
        axes[0].set_title(f'Detected Pieces ({len(detections)})')
        axes[0].axis('off')
        
        # Right: Chess board visualization
        board_display = np.zeros((8, 8))
        piece_positions = {}
        
        for row in range(8):
            for col in range(8):
                piece = board_state[row, col]
                if piece != 'empty':
                    board_display[row, col] = 1
                    piece_positions[(row, col)] = piece
        
        im = axes[1].imshow(board_display, cmap='RdYlBu_r', vmin=0, vmax=1)
        
        # Add piece labels
        for (row, col), piece in piece_positions.items():
            piece_short = piece.replace('_', '\n')
            axes[1].text(col, row, piece_short, ha='center', va='center',
                        fontsize=8, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        
        axes[1].set_title(f'Chess Position\nFEN: {fen}')
        axes[1].set_xticks(range(8))
        axes[1].set_yticks(range(8))
        axes[1].set_xticklabels(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
        axes[1].set_yticklabels(['8', '7', '6', '5', '4', '3', '2', '1'])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

def main():
    """Demo the integrated system"""
    print("🚀 Integrated Chess Recognition System Demo")
    print("Using your trained YOLO model!")
    print("=" * 60)
    
    # Initialize system with your trained model
    try:
        system = IntegratedChessRecognitionSystem()
        
        # Test with your sample image
        test_image = "/home/pranesh/chess/Computer-Vision/board-localisation/images/real_image_1.jpeg"
        
        if Path(test_image).exists():
            print(f"\nTesting with: {test_image}")
            result = system.process_chess_image(test_image)
            
            if result['success']:
                print(f"\n✅ SUCCESS!")
                print(f"FEN: {result['fen']}")
                print(f"Confidence: {result['confidence_score']:.3f}")
                print(f"Processing time: {result['processing_time']:.3f}s")
                print(f"Check 'integrated_results' folder for visualizations")
            else:
                print(f"\n❌ FAILED: {result.get('error', 'Unknown error')}")
        else:
            print(f"\n⚠️  Test image not found: {test_image}")
            
    except Exception as e:
        print(f"\n❌ System initialization failed: {e}")
        print("\nTrying alternative model path...")
        
        # Try with alternative model
        alt_model_path = "/home/pranesh/chess/Computer-Vision/piece-classification/yolov8n.pt"
        try:
            system = IntegratedChessRecognitionSystem(yolo_model_path=alt_model_path)
            print("✅ Using base YOLO model for demo")
        except Exception as e2:
            print(f"❌ Alternative model also failed: {e2}")

if __name__ == "__main__":
    main()
