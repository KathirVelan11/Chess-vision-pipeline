#!/usr/bin/env python3
"""
Simple test to check if all imports work
"""

import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    print("Testing imports...")
    
    print("1. Testing basic imports...")
    import cv2
    import numpy as np
    import torch
    print("   ✓ Basic imports successful")
    
    print("2. Testing YOLO import...")
    from ultralytics import YOLO
    print("   ✓ YOLO import successful")
    
    print("3. Testing utility imports...")
    import utils
    print("   ✓ Utils import successful")
    
    print("4. Testing chess recognition system...")
    import chess_recognition_system
    print("   ✓ Chess recognition system import successful")
    
    print("\n✅ All imports successful!")
    
    # Test configuration loading
    print("\n5. Testing configuration...")
    config = chess_recognition_system.load_config('config.json')
    print(f"   ✓ Configuration loaded: {len(config)} settings")
    
    print("\n6. Testing model path detection...")
    from utils import ModelPathFinder
    piece_model = ModelPathFinder.find_piece_model()
    occupancy_model = ModelPathFinder.find_occupancy_model()
    print(f"   Piece model: {piece_model}")
    print(f"   Occupancy model: {occupancy_model}")
    
    print("\n🎉 System ready for testing!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
