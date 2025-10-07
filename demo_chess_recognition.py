"""
Comprehensive Demo for Chess Recognition System
Based on "Determining Chess Game State From an Image" by Wölflin & Arandjelović

This demo showcases the complete chess recognition pipeline including:
1. Board Localization with RANSAC-based homography
2. Occupancy Classification with ResNet
3. Piece Classification with InceptionV3
4. FEN Generation
5. Transfer Learning for new chess sets

Usage:
    python demo_chess_recognition.py --image path/to/chess/image.jpg
    python demo_chess_recognition.py --batch path/to/images/folder/
    python demo_chess_recognition.py --transfer white_view.jpg black_view.jpg
"""

import argparse
import cv2
import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import json
import time
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from chess_recognition_pipeline import ChessRecognitionPipeline
from transfer_learning_system import ChessTransferLearningSystem
from board_localization_improved import test_board_localization
from fen_generator import FENGenerator

class ChessRecognitionDemo:
    """Comprehensive demo system for chess recognition"""
    
    def __init__(self):
        self.pipeline = None
        self.transfer_system = None
        
        print("🎯 Chess Recognition System Demo")
        print("Based on 'Determining Chess Game State From an Image'")
        print("by Wölflin & Arandjelović")
        print("=" * 60)
    
    def run_single_image_demo(self, image_path: str, output_dir: str = "demo_results"):
        """Demo with a single chess board image"""
        print(f"\n🖼️  SINGLE IMAGE DEMO")
        print(f"Image: {image_path}")
        print("-" * 40)
        
        if not Path(image_path).exists():
            print(f"❌ Image not found: {image_path}")
            return
        
        # Initialize pipeline
        if self.pipeline is None:
            print("🚀 Initializing chess recognition pipeline...")
            self.pipeline = ChessRecognitionPipeline()
        
        # Process image
        start_time = time.time()
        result = self.pipeline.process_image(
            image_path, 
            save_intermediate=True,
            output_dir=output_dir
        )
        
        # Display results
        self._display_single_result(result, time.time() - start_time)
        
        # Save detailed results
        self._save_detailed_results(result, f"{output_dir}/detailed_results.json")
        
        return result
    
    def run_batch_demo(self, images_dir: str, output_dir: str = "batch_demo_results"):
        """Demo with multiple chess board images"""
        print(f"\n📁 BATCH PROCESSING DEMO")
        print(f"Images directory: {images_dir}")
        print("-" * 40)
        
        # Find all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_paths = []
        
        images_path = Path(images_dir)
        if images_path.is_file():
            # Single file provided
            image_paths = [str(images_path)]
        elif images_path.is_dir():
            # Directory provided
            for ext in image_extensions:
                image_paths.extend(list(images_path.glob(f"*{ext}")) + list(images_path.glob(f"*{ext.upper()}")))
            image_paths = [str(p) for p in image_paths]
        
        if not image_paths:
            print(f"❌ No images found in: {images_dir}")
            return
        
        print(f"Found {len(image_paths)} images")
        
        # Initialize pipeline
        if self.pipeline is None:
            print("🚀 Initializing chess recognition pipeline...")
            self.pipeline = ChessRecognitionPipeline()
        
        # Process all images
        results = self.pipeline.batch_process(image_paths, output_dir)
        
        # Display batch summary
        self._display_batch_summary(results)
        
        # Save batch results
        self._save_batch_results(results, f"{output_dir}/batch_results.json")
        
        return results
    
    def run_transfer_learning_demo(self, white_image: str, black_image: str, 
                                 output_dir: str = "transfer_demo_results"):
        """Demo transfer learning with two starting position images"""
        print(f"\n🔄 TRANSFER LEARNING DEMO")
        print(f"White perspective: {white_image}")
        print(f"Black perspective: {black_image}")
        print("-" * 40)
        
        if not Path(white_image).exists():
            print(f"❌ White perspective image not found: {white_image}")
            return
        
        if not Path(black_image).exists():
            print(f"❌ Black perspective image not found: {black_image}")
            return
        
        # Initialize systems
        if self.pipeline is None:
            print("🚀 Initializing base pipeline...")
            self.pipeline = ChessRecognitionPipeline()
        
        if self.transfer_system is None:
            print("🔄 Initializing transfer learning system...")
            self.transfer_system = ChessTransferLearningSystem({
                'occupancy': self.pipeline.occupancy_system,
                'piece': self.pipeline.piece_system
            })
        
        # Run transfer learning
        print("\n🎯 Starting adaptation process...")
        adaptation_results = self.transfer_system.adapt_to_new_chess_set(
            white_perspective_image=white_image,
            black_perspective_image=black_image,
            output_dir=output_dir,
            occupancy_epochs=20,  # Reduced for demo
            piece_epochs=30       # Reduced for demo
        )
        
        # Display transfer learning results
        self._display_transfer_results(adaptation_results)
        
        # Save transfer results
        self._save_transfer_results(adaptation_results, f"{output_dir}/transfer_results.json")
        
        return adaptation_results
    
    def run_component_tests(self, test_image: str):
        """Test individual components of the pipeline"""
        print(f"\n🔧 COMPONENT TESTING DEMO")
        print(f"Test image: {test_image}")
        print("-" * 40)
        
        if not Path(test_image).exists():
            print(f"❌ Test image not found: {test_image}")
            return
        
        # Test 1: Board Localization
        print("\n1️⃣ Testing Board Localization...")
        try:
            localization_result = test_board_localization(test_image)
            print("✅ Board localization test completed")
        except Exception as e:
            print(f"❌ Board localization test failed: {e}")
        
        # Test 2: FEN Generator
        print("\n2️⃣ Testing FEN Generator...")
        try:
            fen_gen = FENGenerator()
            
            # Test with starting position
            starting_fen = fen_gen.get_starting_position_fen()
            board_state, _ = fen_gen.fen_to_board(starting_fen)
            reconstructed_fen = fen_gen.board_to_fen(board_state)
            
            is_valid, errors = fen_gen.validate_fen(reconstructed_fen)
            
            print(f"✅ FEN generator test completed")
            print(f"   Starting FEN: {starting_fen}")
            print(f"   Reconstructed: {reconstructed_fen}")
            print(f"   Valid: {is_valid}")
        except Exception as e:
            print(f"❌ FEN generator test failed: {e}")
        
        # Test 3: Pipeline Components
        print("\n3️⃣ Testing Pipeline Components...")
        try:
            if self.pipeline is None:
                self.pipeline = ChessRecognitionPipeline()
            
            # Load test image
            image = cv2.imread(test_image)
            
            # Test board localization
            homography, corners, _ = self.pipeline.board_localizer.localize_board(image)
            print(f"   Board localization: {'✅ SUCCESS' if homography is not None else '❌ FAILED'}")
            
            if homography is not None:
                # Test occupancy system
                warped = cv2.warpPerspective(image, homography, (400, 400))
                squares = self.pipeline.occupancy_system.extract_squares_with_context(warped)
                occ_preds, occ_confs = self.pipeline.occupancy_system.predict_occupancy(squares[:8])  # Test with 8 squares
                print(f"   Occupancy classification: ✅ SUCCESS ({len(occ_preds)} predictions)")
                
                # Test piece system
                if len(squares) > 0:
                    piece_preds, piece_confs, piece_names = self.pipeline.piece_system.predict_pieces(squares[:4])  # Test with 4 pieces
                    print(f"   Piece classification: ✅ SUCCESS ({len(piece_preds)} predictions)")
        
        except Exception as e:
            print(f"❌ Pipeline component test failed: {e}")
    
    def _display_single_result(self, result: Dict, total_time: float):
        """Display results for single image processing"""
        print(f"\n📊 PROCESSING RESULTS")
        print(f"{'='*50}")
        
        if result['success']:
            print(f"✅ Status: SUCCESS")
            print(f"🕐 Total time: {total_time:.3f}s")
            print(f"🎯 Confidence: {result['confidence_score']:.3f}")
            print(f"♟️  FEN: {result['fen']}")
            
            # Stage breakdown
            print(f"\n📈 Stage Breakdown:")
            for stage, info in result['stages'].items():
                status = "✅" if info.get('success', False) else "❌"
                time_taken = info.get('time', 0)
                print(f"   {stage.title()}: {status} ({time_taken:.3f}s)")
            
            # Additional info
            if 'occupancy' in result['stages']:
                occupied = result['stages']['occupancy'].get('squares_occupied', 0)
                print(f"   Occupied squares: {occupied}")
            
            if 'piece_classification' in result['stages']:
                pieces = result['stages']['piece_classification'].get('pieces_classified', 0)
                print(f"   Pieces classified: {pieces}")
            
            # Validation
            if result.get('validation_errors'):
                print(f"\n⚠️  Validation warnings:")
                for error in result['validation_errors']:
                    print(f"   • {error}")
        else:
            print(f"❌ Status: FAILED")
            print(f"❌ Error: {result.get('error', 'Unknown error')}")
            print(f"🕐 Time before failure: {total_time:.3f}s")
    
    def _display_batch_summary(self, results: List[Dict]):
        """Display summary for batch processing"""
        successful = sum(1 for r in results if r['success'])
        total = len(results)
        
        print(f"\n📊 BATCH PROCESSING SUMMARY")
        print(f"{'='*50}")
        print(f"✅ Successful: {successful}/{total} ({successful/total:.1%})")
        
        if successful > 0:
            # Calculate averages for successful results
            avg_time = np.mean([r['processing_time'] for r in results if r['success']])
            avg_confidence = np.mean([r['confidence_score'] for r in results if r['success']])
            
            print(f"🕐 Average time: {avg_time:.3f}s")
            print(f"🎯 Average confidence: {avg_confidence:.3f}")
            
            # FEN samples
            print(f"\n♟️  Sample FEN results:")
            for i, result in enumerate([r for r in results if r['success']][:3]):
                image_name = Path(result['image_path']).name
                print(f"   {i+1}. {image_name}: {result['fen']}")
        
        # Error summary
        failed = [r for r in results if not r['success']]
        if failed:
            print(f"\n❌ Failed images:")
            for result in failed[:3]:  # Show first 3 failures
                image_name = Path(result['image_path']).name
                error = result.get('error', 'Unknown error')
                print(f"   • {image_name}: {error}")
    
    def _display_transfer_results(self, results: Dict):
        """Display transfer learning results"""
        print(f"\n📊 TRANSFER LEARNING RESULTS")
        print(f"{'='*50}")
        
        if results['success']:
            print(f"✅ Status: SUCCESS")
            
            # Occupancy adaptation
            occ_info = results.get('occupancy_adaptation', {})
            if occ_info.get('success'):
                print(f"🎯 Occupancy adaptation: ✅ SUCCESS")
                print(f"   Training samples: {occ_info.get('training_samples', 0)}")
                print(f"   Epochs: {occ_info.get('epochs', 0)}")
            
            # Piece adaptation  
            piece_info = results.get('piece_adaptation', {})
            if piece_info.get('success'):
                print(f"♟️  Piece adaptation: ✅ SUCCESS")
                print(f"   Training samples: {piece_info.get('training_samples', 0)}")
                print(f"   Epochs: {piece_info.get('epochs', 0)}")
            
            # Validation results
            val_info = results.get('validation_results', {})
            if val_info:
                success_rate = val_info.get('success_rate', 0)
                fen_accuracy = val_info.get('fen_accuracy', 0)
                avg_conf = val_info.get('average_confidence', 0)
                
                print(f"\n🔍 Validation Results:")
                print(f"   Success rate: {success_rate:.1%}")
                print(f"   FEN accuracy: {fen_accuracy:.1%}")
                print(f"   Average confidence: {avg_conf:.3f}")
        else:
            print(f"❌ Status: FAILED")
            print(f"❌ Error: {results.get('error', 'Unknown error')}")
    
    def _save_detailed_results(self, result: Dict, save_path: str):
        """Save detailed results to JSON"""
        # Convert numpy arrays to lists for JSON serialization
        json_result = self._convert_for_json(result)
        
        with open(save_path, 'w') as f:
            json.dump(json_result, f, indent=2)
        
        print(f"💾 Detailed results saved to: {save_path}")
    
    def _save_batch_results(self, results: List[Dict], save_path: str):
        """Save batch results to JSON"""
        json_results = [self._convert_for_json(r) for r in results]
        
        with open(save_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"💾 Batch results saved to: {save_path}")
    
    def _save_transfer_results(self, results: Dict, save_path: str):
        """Save transfer learning results to JSON"""
        json_results = self._convert_for_json(results)
        
        with open(save_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"💾 Transfer results saved to: {save_path}")
    
    def _convert_for_json(self, obj):
        """Convert numpy arrays and other non-serializable objects for JSON"""
        if isinstance(obj, dict):
            return {k: self._convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif hasattr(obj, '__dict__'):
            return str(obj)
        else:
            return obj

def main():
    """Main demo function"""
    parser = argparse.ArgumentParser(description='Chess Recognition System Demo')
    parser.add_argument('--image', type=str, help='Single image to process')
    parser.add_argument('--batch', type=str, help='Directory or pattern for batch processing')
    parser.add_argument('--transfer', nargs=2, metavar=('WHITE_IMG', 'BLACK_IMG'),
                       help='Two starting position images for transfer learning')
    parser.add_argument('--test-components', type=str, help='Test individual components with image')
    parser.add_argument('--output', type=str, default='demo_results', help='Output directory')
    
    args = parser.parse_args()
    
    # Initialize demo system
    demo = ChessRecognitionDemo()
    
    if args.image:
        # Single image demo
        demo.run_single_image_demo(args.image, args.output)
    
    elif args.batch:
        # Batch processing demo
        demo.run_batch_demo(args.batch, args.output)
    
    elif args.transfer:
        # Transfer learning demo
        white_img, black_img = args.transfer
        demo.run_transfer_learning_demo(white_img, black_img, args.output)
    
    elif args.test_components:
        # Component testing
        demo.run_component_tests(args.test_components)
    
    else:
        # Default: run with sample image if available
        sample_image = "/home/pranesh/chess/Computer-Vision/board-localisation/images/real_image_1.jpeg"
        
        if Path(sample_image).exists():
            print("🎯 Running default demo with sample image...")
            demo.run_single_image_demo(sample_image, args.output)
            
            print("\n🔧 Running component tests...")
            demo.run_component_tests(sample_image)
        else:
            print("ℹ️  No arguments provided and no sample image found.")
            print("Usage examples:")
            print("  python demo_chess_recognition.py --image path/to/chess/image.jpg")
            print("  python demo_chess_recognition.py --batch path/to/images/")
            print("  python demo_chess_recognition.py --transfer white.jpg black.jpg")
            print("  python demo_chess_recognition.py --test-components image.jpg")

if __name__ == "__main__":
    main()
