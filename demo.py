#!/usr/bin/env python3
"""
Chess Recognition Demo
=====================

Simple demo script showing how to use the integrated chess recognition system.
This script demonstrates the complete pipeline from image to FEN notation.

Usage:
    python demo.py --image path/to/chess/image.jpg
    python demo.py --help

Author: GitHub Copilot
Date: October 8, 2025
"""

import sys
import os
import argparse
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from chess_recognition_system import ChessRecognitionSystem, load_config
from utils import setup_logging, ModelPathFinder, ConfigManager

def main():
    """Main demo function"""
    parser = argparse.ArgumentParser(
        description='Chess Recognition Demo - Convert chess board images to FEN notation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python demo.py --image board-localisation/images/chess_image_1.jpg
    python demo.py --image my_chess_position.jpg --output result.json
    python demo.py --create-config
        """
    )
    
    parser.add_argument('--image', type=str, 
                       help='Path to chess board image')
    parser.add_argument('--config', type=str, default='config.json',
                       help='Configuration file path (default: config.json)')
    parser.add_argument('--output', type=str,
                       help='Output file for results (JSON format)')
    parser.add_argument('--create-config', action='store_true',
                       help='Create a default configuration file and exit')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--no-visualization', action='store_true',
                       help='Skip visualization display')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = 'DEBUG' if args.verbose else 'INFO'
    setup_logging(log_level)
    
    # Create config if requested
    if args.create_config:
        print("Creating default configuration file...")
        config = ConfigManager.create_default_config(args.config)
        print(f"✓ Configuration saved to: {args.config}")
        print(f"  Found piece model: {config.get('piece_model_path', 'Not found')}")
        print(f"  Found occupancy model: {config.get('occupancy_model_path', 'Not found')}")
        return 0
    
    # Check if image is provided
    if not args.image:
        # Try to find a sample image
        sample_images = [
            'board-localisation/images/chess_image_1.jpg',
            'board-localisation/images/real_image_1.jpeg'
        ]
        
        for sample in sample_images:
            if os.path.exists(sample):
                args.image = sample
                print(f"Using sample image: {sample}")
                break
        
        if not args.image:
            print("❌ No image provided and no sample images found.")
            print("Usage: python demo.py --image path/to/your/chess/image.jpg")
            print("Or run: python demo.py --create-config")
            return 1
    
    # Check if image file exists
    if not os.path.exists(args.image):
        print(f"❌ Image file not found: {args.image}")
        return 1
    
    print(f"""
{'='*60}
🏰 CHESS RECOGNITION SYSTEM DEMO
{'='*60}
Image: {args.image}
Config: {args.config}
{'='*60}
""")
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Auto-detect models if not specified
        if not config.get('piece_model_path'):
            piece_model = ModelPathFinder.find_piece_model()
            if piece_model:
                config['piece_model_path'] = piece_model
                print(f"🔍 Auto-detected piece model: {piece_model}")
            else:
                print("⚠️  No piece classification model found")
        
        if not config.get('occupancy_model_path'):
            occupancy_model = ModelPathFinder.find_occupancy_model()
            if occupancy_model:
                config['occupancy_model_path'] = occupancy_model
                print(f"🔍 Auto-detected occupancy model: {occupancy_model}")
            else:
                print("⚠️  No occupancy classification model found (will use piece detection only)")
        
        # Initialize recognition system
        print("🚀 Initializing recognition system...")
        system = ChessRecognitionSystem(config)
        
        # Process the image
        print("🔄 Processing image...")
        result = system.recognize_chess_state(args.image)
        
        if result['success']:
            print(f"""
✅ RECOGNITION SUCCESSFUL!

📋 FEN Notation:
{result['fen']}

📊 Statistics:
   • Total squares: {result['statistics']['total_squares']}
   • Occupied squares: {result['statistics']['occupied_squares']}
   • Identified pieces: {result['statistics']['identified_pieces']}
   • Average confidence: {result['statistics']['average_confidence']:.3f}
   • Recognition rate: {result['statistics']['recognition_rate']:.1%}

🏰 Detected Pieces:""")
            
            # Show piece distribution
            piece_counts = {}
            for square in result['squares']:
                if square['piece']:
                    piece = square['piece']
                    piece_counts[piece] = piece_counts.get(piece, 0) + 1
            
            if piece_counts:
                for piece, count in sorted(piece_counts.items()):
                    print(f"   • {piece}: {count}")
            else:
                print("   • No pieces detected")
            
            # Save results if requested
            if args.output:
                import json
                with open(args.output, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"\n💾 Results saved to: {args.output}")
            
            # Show visualization if not disabled
            if not args.no_visualization:
                try:
                    print(f"\n🖼️  Displaying visualization...")
                    from test_chess_recognition import ChessRecognitionTester
                    tester = ChessRecognitionTester(args.config)
                    tester.system = system  # Reuse the system
                    tester._visualize_results(args.image, result)
                except Exception as e:
                    print(f"⚠️  Visualization failed: {e}")
            
            print(f"\n🎉 Demo completed successfully!")
            
        else:
            print(f"""
❌ RECOGNITION FAILED!

Error: {result['error']}

💡 Troubleshooting tips:
   • Ensure the image contains a clear view of a chess board
   • Check that the board is well-lit and not too blurry
   • Verify that the trained models are available
   • Try adjusting the configuration parameters
""")
            return 1
            
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
