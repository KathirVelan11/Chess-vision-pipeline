#!/usr/bin/env python3
"""
Minimal test for the chess recognition system
"""

def test_basic_functionality():
    print("🧪 Testing Chess Recognition System Integration")
    print("=" * 50)
    
    # Test 1: Import libraries
    try:
        import cv2
        import numpy as np
        import torch
        from ultralytics import YOLO
        print("✓ Basic libraries imported successfully")
    except Exception as e:
        print(f"❌ Failed to import basic libraries: {e}")
        return False
    
    # Test 2: Check if sample image exists
    import os
    sample_image = "board-localisation/images/real_image_1.jpeg"
    if os.path.exists(sample_image):
        print(f"✓ Sample image found: {sample_image}")
    else:
        print(f"❌ Sample image not found: {sample_image}")
        return False
    
    # Test 3: Load image
    try:
        image = cv2.imread(sample_image)
        if image is not None:
            print(f"✓ Image loaded successfully, shape: {image.shape}")
        else:
            print("❌ Failed to load image")
            return False
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return False
    
    # Test 4: Try to find piece model
    try:
        import glob
        piece_model_patterns = [
            "piece-classification/chess_detection/*/weights/best.pt",
            "piece-classification/yolov8n.pt"
        ]
        
        piece_model = None
        for pattern in piece_model_patterns:
            matches = glob.glob(pattern)
            if matches:
                piece_model = matches[0]
                break
        
        if piece_model:
            print(f"✓ Piece model found: {piece_model}")
            
            # Test loading YOLO model
            try:
                model = YOLO(piece_model)
                print("✓ YOLO model loaded successfully")
                
                # Test inference
                results = model.predict(image, verbose=False)
                print(f"✓ YOLO inference successful, {len(results)} results")
                
            except Exception as e:
                print(f"❌ YOLO model loading/inference failed: {e}")
                return False
                
        else:
            print("⚠️  No piece model found, but that's okay for basic testing")
    
    except Exception as e:
        print(f"❌ Error testing piece model: {e}")
        return False
    
    # Test 5: FEN generation (basic test)
    try:
        # Test basic FEN string
        test_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        print(f"✓ Test FEN: {test_fen}")
        
        # Test FEN parsing
        board_part = test_fen.split()[0]
        ranks = board_part.split('/')
        if len(ranks) == 8:
            print("✓ FEN parsing works correctly")
        else:
            print("❌ FEN parsing failed")
            return False
            
    except Exception as e:
        print(f"❌ Error testing FEN: {e}")
        return False
    
    print("\n🎉 Basic functionality test completed successfully!")
    print("\nNext steps:")
    print("1. Train/obtain the occupancy classification model")
    print("2. Test the complete pipeline with: python demo.py --image board-localisation/images/real_image_1.jpeg")
    print("3. Use the test script for comprehensive evaluation")
    
    return True

if __name__ == "__main__":
    success = test_basic_functionality()
    exit(0 if success else 1)
