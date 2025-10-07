"""
Test script for the Chess Recognition System
===========================================

This script tests the integrated chess recognition system and provides
visualization tools to understand the recognition process.

Author: GitHub Copilot
Date: October 8, 2025
"""

import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from chess_recognition_system import ChessRecognitionSystem, load_config

class ChessRecognitionTester:
    """Test and visualization utilities for chess recognition"""
    
    def __init__(self, config_path: str = None):
        """Initialize the tester with configuration"""
        self.config = load_config(config_path)
        self.system = None
        
    def initialize_system(self):
        """Initialize the recognition system"""
        try:
            self.system = ChessRecognitionSystem(self.config)
            print("✓ Chess Recognition System initialized successfully")
            return True
        except Exception as e:
            print(f"✗ Failed to initialize system: {e}")
            return False
    
    def test_single_image(self, image_path: str, save_results: bool = True, 
                         show_visualization: bool = True) -> Dict:
        """Test recognition on a single image with detailed output"""
        
        if not self.system:
            if not self.initialize_system():
                return {'success': False, 'error': 'System initialization failed'}
        
        print(f"\n{'='*60}")
        print(f"TESTING IMAGE: {image_path}")
        print(f"{'='*60}")
        
        # Check if image exists
        if not os.path.exists(image_path):
            print(f"✗ Image not found: {image_path}")
            return {'success': False, 'error': 'Image not found'}
        
        # Run recognition
        try:
            result = self.system.recognize_chess_state(image_path)
            
            if result['success']:
                print(f"✓ Recognition completed successfully!")
                print(f"📋 FEN: {result['fen']}")
                
                # Print statistics
                stats = result['statistics']
                print(f"\n📊 STATISTICS:")
                print(f"   • Total squares: {stats['total_squares']}")
                print(f"   • Occupied squares: {stats['occupied_squares']}")
                print(f"   • Identified pieces: {stats['identified_pieces']}")
                print(f"   • Average confidence: {stats['average_confidence']:.3f}")
                print(f"   • Recognition rate: {stats['recognition_rate']:.1%}")
                
                # Show piece distribution
                self._print_piece_distribution(result['squares'])
                
                # Save results
                if save_results:
                    self._save_results(result, image_path)
                
                # Show visualization
                if show_visualization:
                    self._visualize_results(image_path, result)
                
            else:
                print(f"✗ Recognition failed: {result['error']}")
            
            return result
            
        except Exception as e:
            print(f"✗ Error during recognition: {e}")
            return {'success': False, 'error': str(e)}
    
    def _print_piece_distribution(self, squares: List[Dict]):
        """Print distribution of detected pieces"""
        piece_counts = {}
        
        for square in squares:
            if square['piece']:
                piece = square['piece']
                piece_counts[piece] = piece_counts.get(piece, 0) + 1
        
        if piece_counts:
            print(f"\n🏰 PIECE DISTRIBUTION:")
            for piece, count in sorted(piece_counts.items()):
                print(f"   • {piece}: {count}")
        else:
            print(f"\n🏰 No pieces detected")
    
    def _save_results(self, result: Dict, image_path: str):
        """Save results to JSON file"""
        output_dir = Path(self.config.get('output_dir', 'results'))
        output_dir.mkdir(exist_ok=True)
        
        # Create filename based on input image
        image_name = Path(image_path).stem
        result_file = output_dir / f"{image_name}_recognition_result.json"
        
        try:
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"💾 Results saved to: {result_file}")
        except Exception as e:
            print(f"✗ Failed to save results: {e}")
    
    def _visualize_results(self, image_path: str, result: Dict):
        """Create comprehensive visualization of the recognition results"""
        try:
            # Load original image
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Create figure with multiple subplots
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(f'Chess Recognition Results: {Path(image_path).name}', fontsize=16)
            
            # 1. Original image
            axes[0, 0].imshow(image)
            axes[0, 0].set_title('Original Image')
            axes[0, 0].axis('off')
            
            # 2. Board corners detection
            corners_img = image.copy()
            if 'board_corners' in result and result['board_corners']:
                corners = result['board_corners']
                # Draw corners
                for i, (x, y) in enumerate(corners):
                    cv2.circle(corners_img, (x, y), 10, (255, 0, 0), -1)
                    cv2.putText(corners_img, str(i), (x+15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                # Draw board outline
                pts = np.array(corners, np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(corners_img, [pts], True, (0, 255, 0), 3)
            
            axes[0, 1].imshow(corners_img)
            axes[0, 1].set_title('Board Corners Detection')
            axes[0, 1].axis('off')
            
            # 3. Board layout visualization
            self._draw_board_layout(axes[0, 2], result['squares'])
            
            # 4. Occupancy heatmap
            self._draw_occupancy_heatmap(axes[1, 0], result['squares'])
            
            # 5. Confidence heatmap
            self._draw_confidence_heatmap(axes[1, 1], result['squares'])
            
            # 6. FEN visualization
            self._draw_fen_board(axes[1, 2], result['fen'])
            
            plt.tight_layout()
            
            # Save visualization
            output_dir = Path(self.config.get('output_dir', 'results'))
            output_dir.mkdir(exist_ok=True)
            image_name = Path(image_path).stem
            viz_file = output_dir / f"{image_name}_visualization.png"
            plt.savefig(viz_file, dpi=300, bbox_inches='tight')
            print(f"📸 Visualization saved to: {viz_file}")
            
            plt.show()
            
        except Exception as e:
            print(f"✗ Visualization failed: {e}")
    
    def _draw_board_layout(self, ax, squares: List[Dict]):
        """Draw the detected board layout"""
        board = np.zeros((8, 8, 3), dtype=np.uint8)
        
        # Color squares based on occupancy and pieces
        for square in squares:
            row, col = square['row'], square['col']
            
            if square['is_occupied']:
                if square['piece'] and 'white' in square['piece']:
                    board[row, col] = [255, 255, 255]  # White piece
                elif square['piece'] and 'black' in square['piece']:
                    board[row, col] = [100, 100, 100]  # Black piece
                else:
                    board[row, col] = [255, 255, 0]    # Unknown piece
            else:
                # Alternating board colors
                if (row + col) % 2 == 0:
                    board[row, col] = [240, 217, 181]  # Light square
                else:
                    board[row, col] = [181, 136, 99]   # Dark square
        
        ax.imshow(board, aspect='equal')
        ax.set_title('Board Layout\n(White=White pieces, Gray=Black pieces)')
        
        # Add grid and labels
        ax.set_xticks(range(8))
        ax.set_yticks(range(8))
        ax.set_xticklabels(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
        ax.set_yticklabels(['8', '7', '6', '5', '4', '3', '2', '1'])
        ax.grid(True, color='black', linewidth=1)
    
    def _draw_occupancy_heatmap(self, ax, squares: List[Dict]):
        """Draw occupancy heatmap"""
        occupancy = np.zeros((8, 8))
        
        for square in squares:
            row, col = square['row'], square['col']
            occupancy[row, col] = 1 if square['is_occupied'] else 0
        
        im = ax.imshow(occupancy, cmap='Reds', aspect='equal', vmin=0, vmax=1)
        ax.set_title('Square Occupancy')
        ax.set_xticks(range(8))
        ax.set_yticks(range(8))
        ax.set_xticklabels(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
        ax.set_yticklabels(['8', '7', '6', '5', '4', '3', '2', '1'])
        
        # Add text annotations
        for square in squares:
            row, col = square['row'], square['col']
            text = 'O' if square['is_occupied'] else 'E'
            ax.text(col, row, text, ha='center', va='center', 
                   color='white' if square['is_occupied'] else 'black', fontweight='bold')
    
    def _draw_confidence_heatmap(self, ax, squares: List[Dict]):
        """Draw confidence heatmap"""
        confidence = np.zeros((8, 8))
        
        for square in squares:
            row, col = square['row'], square['col']
            confidence[row, col] = square['confidence']
        
        im = ax.imshow(confidence, cmap='viridis', aspect='equal', vmin=0, vmax=1)
        ax.set_title('Classification Confidence')
        ax.set_xticks(range(8))
        ax.set_yticks(range(8))
        ax.set_xticklabels(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
        ax.set_yticklabels(['8', '7', '6', '5', '4', '3', '2', '1'])
        
        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Add text annotations
        for square in squares:
            row, col = square['row'], square['col']
            if square['confidence'] > 0:
                ax.text(col, row, f"{square['confidence']:.2f}", 
                       ha='center', va='center', color='white', fontsize=8)
    
    def _draw_fen_board(self, ax, fen: str):
        """Draw board from FEN notation"""
        if not fen:
            ax.text(0.5, 0.5, 'No FEN Available', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=16)
            ax.set_title('FEN Board')
            return
        
        # Parse FEN (only board part)
        board_fen = fen.split()[0]
        ranks = board_fen.split('/')
        
        board = np.ones((8, 8, 3), dtype=np.float32)
        
        # Unicode chess pieces
        piece_symbols = {
            'K': '♔', 'Q': '♕', 'R': '♖', 'B': '♗', 'N': '♘', 'P': '♙',
            'k': '♚', 'q': '♛', 'r': '♜', 'b': '♝', 'n': '♞', 'p': '♟'
        }
        
        # Color squares
        for rank_idx, rank in enumerate(ranks):
            file_idx = 0
            for char in rank:
                if char.isdigit():
                    # Empty squares
                    for _ in range(int(char)):
                        if (rank_idx + file_idx) % 2 == 0:
                            board[rank_idx, file_idx] = [0.95, 0.85, 0.71]  # Light
                        else:
                            board[rank_idx, file_idx] = [0.71, 0.53, 0.39]  # Dark
                        file_idx += 1
                else:
                    # Piece square
                    if (rank_idx + file_idx) % 2 == 0:
                        board[rank_idx, file_idx] = [0.95, 0.85, 0.71]  # Light
                    else:
                        board[rank_idx, file_idx] = [0.71, 0.53, 0.39]  # Dark
                    file_idx += 1
        
        ax.imshow(board, aspect='equal')
        
        # Add pieces
        for rank_idx, rank in enumerate(ranks):
            file_idx = 0
            for char in rank:
                if char.isdigit():
                    file_idx += int(char)
                else:
                    symbol = piece_symbols.get(char, char)
                    color = 'black' if char.islower() else 'white'
                    ax.text(file_idx, rank_idx, symbol, ha='center', va='center',
                           fontsize=20, color=color, 
                           bbox=dict(boxstyle="round,pad=0.1", facecolor='none', edgecolor='none'))
                    file_idx += 1
        
        ax.set_title(f'FEN Board\n{fen}', fontsize=10)
        ax.set_xticks(range(8))
        ax.set_yticks(range(8))
        ax.set_xticklabels(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
        ax.set_yticklabels(['8', '7', '6', '5', '4', '3', '2', '1'])
    
    def test_multiple_images(self, image_dir: str, pattern: str = "*.jpg"):
        """Test recognition on multiple images"""
        from glob import glob
        
        image_paths = glob(os.path.join(image_dir, pattern))
        if not image_paths:
            print(f"No images found in {image_dir} matching {pattern}")
            return
        
        print(f"\n{'='*60}")
        print(f"TESTING MULTIPLE IMAGES: {len(image_paths)} images found")
        print(f"{'='*60}")
        
        results = []
        for i, image_path in enumerate(image_paths, 1):
            print(f"\n[{i}/{len(image_paths)}] Processing: {os.path.basename(image_path)}")
            result = self.test_single_image(image_path, save_results=True, show_visualization=False)
            results.append({
                'image_path': image_path,
                'result': result
            })
        
        # Summary
        successful = sum(1 for r in results if r['result']['success'])
        print(f"\n{'='*60}")
        print(f"BATCH TESTING SUMMARY")
        print(f"{'='*60}")
        print(f"Total images: {len(results)}")
        print(f"Successful: {successful}")
        print(f"Failed: {len(results) - successful}")
        print(f"Success rate: {successful/len(results):.1%}")
        
        return results
    
    def benchmark_performance(self, image_path: str, iterations: int = 5):
        """Benchmark the performance of the recognition system"""
        import time
        
        if not self.system:
            if not self.initialize_system():
                return
        
        print(f"\n{'='*60}")
        print(f"PERFORMANCE BENCHMARK: {iterations} iterations")
        print(f"{'='*60}")
        
        times = []
        for i in range(iterations):
            start_time = time.time()
            result = self.system.recognize_chess_state(image_path)
            end_time = time.time()
            
            processing_time = end_time - start_time
            times.append(processing_time)
            
            status = "✓" if result['success'] else "✗"
            print(f"Iteration {i+1}: {processing_time:.3f}s {status}")
        
        # Statistics
        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        
        print(f"\n📊 PERFORMANCE STATISTICS:")
        print(f"   • Average time: {avg_time:.3f}s ± {std_time:.3f}s")
        print(f"   • Min time: {min_time:.3f}s")
        print(f"   • Max time: {max_time:.3f}s")
        print(f"   • Throughput: {1/avg_time:.1f} images/second")

def main():
    """Main function for command-line testing"""
    parser = argparse.ArgumentParser(description='Test Chess Recognition System')
    parser.add_argument('--image', type=str, help='Path to single image to test')
    parser.add_argument('--image-dir', type=str, help='Directory with multiple images to test')
    parser.add_argument('--pattern', type=str, default='*.jpg', help='Pattern for image files (default: *.jpg)')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--benchmark', action='store_true', help='Run performance benchmark')
    parser.add_argument('--iterations', type=int, default=5, help='Number of benchmark iterations')
    parser.add_argument('--no-visualization', action='store_true', help='Skip visualization display')
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = ChessRecognitionTester(args.config)
    
    if args.image:
        # Test single image
        show_viz = not args.no_visualization
        result = tester.test_single_image(args.image, show_visualization=show_viz)
        
        if args.benchmark and result['success']:
            tester.benchmark_performance(args.image, args.iterations)
            
    elif args.image_dir:
        # Test multiple images
        tester.test_multiple_images(args.image_dir, args.pattern)
        
    else:
        # Default test with sample image
        sample_images = [
            'board-localisation/images/chess_image_1.jpg',
            'board-localisation/images/real_image_1.jpeg'
        ]
        
        for sample_image in sample_images:
            if os.path.exists(sample_image):
                print(f"Testing with sample image: {sample_image}")
                tester.test_single_image(sample_image, show_visualization=not args.no_visualization)
                break
        else:
            print("No sample images found. Please provide --image or --image-dir argument.")

if __name__ == "__main__":
    main()
