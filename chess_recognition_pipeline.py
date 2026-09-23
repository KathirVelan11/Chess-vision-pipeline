"""
Complete Chess Recognition Pipeline
Based on "Determining Chess Game State From an Image" by Wölflin & Arandjelović

This is the main end-to-end pipeline that integrates:
1. Board Localization (Section 4.1)
2. Occupancy Classification (Section 4.2) 
3. Piece Classification (Section 4.3)
4. FEN Generation

The pipeline takes a single chess board image and outputs FEN notation.
"""

import cv2
import numpy as np
import torch
import time
from typing import Tuple, Optional, Dict, List
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from board_localization_improved import ChessBoardLocalizer
from occupancy_classification_improved import ChessOccupancySystem
from piece_classification_improved import ChessPieceSystem
from fen_generator import FENGenerator

class ChessRecognitionPipeline:
    """Complete chess recognition pipeline"""
    
    def __init__(self, 
                 occupancy_model_path: str = None,
                 piece_model_path: str = None,
                 device: str = 'cuda'):
        """
        Initialize the chess recognition pipeline
        
        Args:
            occupancy_model_path: Path to trained occupancy model
            piece_model_path: Path to trained piece model  
            device: Computing device ('cuda' or 'cpu')
        """
        self.device = device
        
        # Initialize components
        self.board_localizer = ChessBoardLocalizer()
        self.occupancy_system = ChessOccupancySystem(occupancy_model_path, device)
        self.piece_system = ChessPieceSystem(piece_model_path, device)
        self.fen_generator = FENGenerator()
        
        print(f"Chess Recognition Pipeline initialized on {device}")
        print(f"- Board Localizer: Ready")
        print(f"- Occupancy System: {'Model loaded' if occupancy_model_path else 'Using pretrained features'}")
        print(f"- Piece System: {'Model loaded' if piece_model_path else 'Using pretrained features'}")
        print(f"- FEN Generator: Ready")
    
    def process_image(self, image_path: str, 
                     save_intermediate: bool = True,
                     output_dir: str = "pipeline_results") -> Dict:
        """
        Process a single chess board image through the complete pipeline
        
        Args:
            image_path: Path to input chess board image
            save_intermediate: Whether to save intermediate results
            output_dir: Directory to save results
            
        Returns:
            Dictionary containing all pipeline results
        """
        start_time = time.time()
        
        # Create output directory
        if save_intermediate:
            Path(output_dir).mkdir(exist_ok=True)
        
        results = {
            'image_path': image_path,
            'success': False,
            'fen': None,
            'confidence_score': 0.0,
            'processing_time': 0.0,
            'stages': {}
        }
        
        try:
            # Load and preprocess image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            print(f"\nProcessing: {image_path}")
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
            
            print(f"Board localization: {'SUCCESS' if homography is not None else 'FAILED'}")
            print(f"Processing time: {stage1_time:.3f}s")
            print(f"Corner points detected: {len(corner_points)}")
            
            if homography is None:
                print("❌ Board localization failed - cannot proceed")
                return results
            
            # Rectify the board using homography
            board_size = 400  # Standard size for processing
            warped_board = cv2.warpPerspective(image, homography, (board_size, board_size))
            
            if save_intermediate:
                cv2.imwrite(f"{output_dir}/01_warped_board.png", warped_board)
            
            # Stage 2: Occupancy Classification
            print("\n" + "="*50)
            print("STAGE 2: OCCUPANCY CLASSIFICATION")
            print("="*50)
            
            stage2_start = time.time()
            
            # Extract squares with 50% context
            squares = self.occupancy_system.extract_squares_with_context(warped_board, context_factor=0.5)
            
            # Predict occupancy
            occupancy_predictions, occupancy_confidences = self.occupancy_system.predict_occupancy(squares)
            stage2_time = time.time() - stage2_start
            
            # Convert predictions to 8x8 grid
            occupancy_grid = np.array(occupancy_predictions).reshape(8, 8)
            confidence_grid = np.array(occupancy_confidences).reshape(8, 8)
            
            num_occupied = np.sum(occupancy_grid)
            avg_confidence = np.mean(occupancy_confidences)
            
            results['stages']['occupancy'] = {
                'success': len(occupancy_predictions) == 64,
                'time': stage2_time,
                'squares_occupied': int(num_occupied),
                'average_confidence': float(avg_confidence)
            }
            
            print(f"Occupancy classification: {'SUCCESS' if len(occupancy_predictions) == 64 else 'FAILED'}")
            print(f"Processing time: {stage2_time:.3f}s")
            print(f"Occupied squares: {num_occupied}/64")
            print(f"Average confidence: {avg_confidence:.3f}")
            
            if save_intermediate:
                self._save_occupancy_visualization(warped_board, occupancy_grid, confidence_grid, 
                                                 f"{output_dir}/02_occupancy_results.png")
            
            # Stage 3: Piece Classification
            print("\n" + "="*50)
            print("STAGE 3: PIECE CLASSIFICATION")
            print("="*50)
            
            stage3_start = time.time()
            
            # Extract pieces from occupied squares
            piece_images, piece_positions = self._extract_occupied_pieces(warped_board, occupancy_grid)
            
            if len(piece_images) > 0:
                # Predict piece types
                piece_predictions, piece_confidences, piece_class_names = self.piece_system.predict_pieces(piece_images)
                stage3_time = time.time() - stage3_start
                
                avg_piece_confidence = np.mean(piece_confidences) if piece_confidences else 0.0
                
                results['stages']['piece_classification'] = {
                    'success': len(piece_predictions) == len(piece_images),
                    'time': stage3_time,
                    'pieces_classified': len(piece_predictions),
                    'average_confidence': float(avg_piece_confidence)
                }
                
                print(f"Piece classification: {'SUCCESS' if len(piece_predictions) == len(piece_images) else 'FAILED'}")
                print(f"Processing time: {stage3_time:.3f}s")
                print(f"Pieces classified: {len(piece_predictions)}")
                print(f"Average confidence: {avg_piece_confidence:.3f}")
                
                if save_intermediate:
                    self._save_piece_visualization(piece_images, piece_class_names, piece_confidences,
                                                 f"{output_dir}/03_piece_results.png")
            else:
                piece_predictions = []
                piece_confidences = []
                piece_class_names = []
                stage3_time = time.time() - stage3_start
                
                results['stages']['piece_classification'] = {
                    'success': True,
                    'time': stage3_time,
                    'pieces_classified': 0,
                    'average_confidence': 1.0
                }
                
                print("No occupied squares detected - empty board")
            
            # Stage 4: FEN Generation
            print("\n" + "="*50)
            print("STAGE 4: FEN GENERATION")
            print("="*50)
            
            stage4_start = time.time()
            
            # Create board state from predictions
            board_state = self.fen_generator.create_board_from_predictions(
                occupancy_grid, piece_class_names, piece_positions
            )
            
            # Generate FEN notation
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
            
            print(f"FEN generation: SUCCESS")
            print(f"Processing time: {stage4_time:.3f}s")
            print(f"Generated FEN: {fen}")
            print(f"FEN valid: {is_valid}")
            
            if validation_errors:
                print(f"Validation warnings: {validation_errors}")
            
            if save_intermediate:
                self._save_final_visualization(warped_board, board_state, fen, 
                                             f"{output_dir}/04_final_result.png")
            
            # Calculate overall confidence score
            confidence_components = [
                avg_confidence,  # Occupancy confidence
                avg_piece_confidence if piece_confidences else 1.0,  # Piece confidence
                1.0 if is_valid else 0.5  # FEN validity bonus
            ]
            overall_confidence = np.mean(confidence_components)
            
            # Final results
            total_time = time.time() - start_time
            
            results.update({
                'success': True,
                'fen': fen,
                'confidence_score': float(overall_confidence),
                'processing_time': total_time,
                'board_state': board_state.tolist(),
                'occupancy_grid': occupancy_grid.tolist(),
                'piece_predictions': piece_class_names,
                'validation_errors': validation_errors
            })
            
            print(f"\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
            print(f"Total processing time: {total_time:.3f}s")
            print(f"Overall confidence: {overall_confidence:.3f}")
            print(f"Final FEN: {fen}")
            
        except Exception as e:
            print(f"\n❌ Pipeline failed with error: {str(e)}")
            results.update({
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time
            })
        
        return results
    
    def _extract_occupied_pieces(self, warped_board: np.ndarray, 
                               occupancy_grid: np.ndarray) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
        """Extract piece images from occupied squares with proper bounding boxes"""
        pieces = self.piece_system.extract_piece_bounding_boxes(warped_board, occupancy_grid)
        
        if pieces:
            piece_images, positions = zip(*pieces)
            return list(piece_images), list(positions)
        else:
            return [], []
    
    def _save_occupancy_visualization(self, warped_board: np.ndarray, 
                                    occupancy_grid: np.ndarray,
                                    confidence_grid: np.ndarray,
                                    save_path: str):
        """Save occupancy classification visualization"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original warped board
        axes[0].imshow(cv2.cvtColor(warped_board, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Rectified Board')
        axes[0].axis('off')
        
        # Occupancy grid
        im1 = axes[1].imshow(occupancy_grid, cmap='RdYlBu_r', vmin=0, vmax=1)
        axes[1].set_title(f'Occupancy Predictions\n{np.sum(occupancy_grid)} occupied squares')
        axes[1].set_xticks(range(8))
        axes[1].set_yticks(range(8))
        axes[1].set_xticklabels(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
        axes[1].set_yticklabels(['8', '7', '6', '5', '4', '3', '2', '1'])
        plt.colorbar(im1, ax=axes[1], label='Occupied')
        
        # Confidence grid
        im2 = axes[2].imshow(confidence_grid, cmap='viridis', vmin=0, vmax=1)
        axes[2].set_title(f'Confidence Scores\nMean: {np.mean(confidence_grid):.3f}')
        axes[2].set_xticks(range(8))
        axes[2].set_yticks(range(8))
        axes[2].set_xticklabels(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
        axes[2].set_yticklabels(['8', '7', '6', '5', '4', '3', '2', '1'])
        plt.colorbar(im2, ax=axes[2], label='Confidence')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _save_piece_visualization(self, piece_images: List[np.ndarray],
                                piece_class_names: List[str],
                                piece_confidences: List[float],
                                save_path: str):
        """Save piece classification visualization"""
        num_pieces = len(piece_images)
        if num_pieces == 0:
            return
        
        cols = min(8, num_pieces)
        rows = (num_pieces + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(2*cols, 2*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        for i in range(num_pieces):
            row = i // cols
            col = i % cols
            
            if len(piece_images[i].shape) == 3:
                axes[row, col].imshow(cv2.cvtColor(piece_images[i], cv2.COLOR_BGR2RGB))
            else:
                axes[row, col].imshow(piece_images[i], cmap='gray')
            
            axes[row, col].set_title(f'{piece_class_names[i]}\n{piece_confidences[i]:.3f}', fontsize=8)
            axes[row, col].axis('off')
        
        # Hide empty subplots
        for i in range(num_pieces, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _save_final_visualization(self, warped_board: np.ndarray,
                                board_state: np.ndarray,
                                fen: str,
                                save_path: str):
        """Save final result visualization"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Warped board
        axes[0].imshow(cv2.cvtColor(warped_board, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Rectified Chess Board')
        axes[0].axis('off')
        
        # Board state visualization
        board_display = np.zeros((8, 8))
        piece_symbols = {}
        
        for row in range(8):
            for col in range(8):
                piece = board_state[row, col]
                if piece != 'empty' and piece is not None:
                    board_display[row, col] = 1
                    piece_symbols[(row, col)] = piece
        
        im = axes[1].imshow(board_display, cmap='RdYlBu_r', vmin=0, vmax=1)
        
        # Add piece labels
        for (row, col), piece in piece_symbols.items():
            piece_short = piece.replace('_', ' ').title()
            axes[1].text(col, row, piece_short, ha='center', va='center', 
                        fontsize=6, fontweight='bold', 
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        
        axes[1].set_title(f'Chess Position\nFEN: {fen}')
        axes[1].set_xticks(range(8))
        axes[1].set_yticks(range(8))
        axes[1].set_xticklabels(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
        axes[1].set_yticklabels(['8', '7', '6', '5', '4', '3', '2', '1'])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def batch_process(self, image_paths: List[str], 
                     output_dir: str = "batch_results") -> List[Dict]:
        """Process multiple images"""
        results = []
        
        print(f"\nProcessing {len(image_paths)} images...")
        
        for i, image_path in enumerate(image_paths):
            print(f"\n{'='*60}")
            print(f"PROCESSING IMAGE {i+1}/{len(image_paths)}")
            print(f"{'='*60}")
            
            # Create subdirectory for each image
            image_name = Path(image_path).stem
            image_output_dir = Path(output_dir) / image_name
            
            result = self.process_image(image_path, save_intermediate=True, 
                                      output_dir=str(image_output_dir))
            results.append(result)
        
        # Summary
        successful = sum(1 for r in results if r['success'])
        print(f"\n{'='*60}")
        print(f"BATCH PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"Successfully processed: {successful}/{len(image_paths)} images")
        
        avg_time = np.mean([r['processing_time'] for r in results])
        avg_confidence = np.mean([r.get('confidence_score', 0) for r in results if r['success']])
        
        print(f"Average processing time: {avg_time:.3f}s")
        print(f"Average confidence: {avg_confidence:.3f}")
        
        return results

def demo_pipeline():
    """Demonstrate the chess recognition pipeline"""
    print("🚀 Chess Recognition Pipeline Demo")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = ChessRecognitionPipeline()
    
    # Test with sample image
    test_image = "/home/pranesh/chess/Computer-Vision/board-localisation/images/real_image_1.jpeg"
    
    if Path(test_image).exists():
        print(f"\nTesting with image: {test_image}")
        
        # Process single image
        result = pipeline.process_image(test_image, save_intermediate=True)
        
        if result['success']:
            print(f"\n✅ Success! Generated FEN: {result['fen']}")
            print(f"Confidence: {result['confidence_score']:.3f}")
            print(f"Processing time: {result['processing_time']:.3f}s")
        else:
            print(f"\n❌ Failed: {result.get('error', 'Unknown error')}")
    else:
        print(f"\n⚠️  Test image not found: {test_image}")
        print("Please update the path to a valid chess board image.")

if __name__ == "__main__":
    demo_pipeline()
