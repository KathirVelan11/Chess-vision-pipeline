# Chess Recognition System

A complete implementation of the chess recognition system described in "Determining Chess Game State From an Image" by Wölflin & Arandjelović (2021).

## 📋 Overview

This project implements a state-of-the-art chess recognition pipeline that can analyze a photograph of a chess board and output the position in standard FEN notation. The system achieves high accuracy through a three-stage approach:

1. **Board Localization** - RANSAC-based homography computation to detect and rectify the chessboard
2. **Occupancy Classification** - ResNet-based binary classifier to determine which squares are occupied
3. **Piece Classification** - InceptionV3-based 12-class classifier to identify piece types

## 🎯 Key Features

- **End-to-end pipeline** from raw image to FEN notation
- **RANSAC-based board localization** for robust corner detection
- **Deep learning models** with ResNet and InceptionV3 architectures
- **Few-shot transfer learning** to adapt to new chess sets using just 2 images
- **Comprehensive validation** with FEN notation verification
- **Batch processing** capabilities for multiple images
- **Detailed visualization** of intermediate results

## 📊 Performance

Based on the original paper methodology:
- **Per-square accuracy**: 99.77% (0.23% error rate)
- **28x improvement** over previous state-of-the-art
- **Transfer learning accuracy**: 99.83% on new chess sets
- **Processing time**: <0.5s on GPU, ~2s on CPU

## 🚀 Quick Start

### Prerequisites

```bash
pip install torch torchvision opencv-python scikit-learn matplotlib seaborn
pip install albumentations pillow numpy pathlib
```

### Basic Usage

```python
from chess_recognition_pipeline import ChessRecognitionPipeline

# Initialize pipeline
pipeline = ChessRecognitionPipeline()

# Process single image
result = pipeline.process_image("path/to/chess/image.jpg")

if result['success']:
    print(f"FEN: {result['fen']}")
    print(f"Confidence: {result['confidence_score']:.3f}")
```

### Demo Script

```bash
# Single image processing
python demo_chess_recognition.py --image path/to/chess/image.jpg

# Batch processing
python demo_chess_recognition.py --batch path/to/images/directory/

# Transfer learning (adapt to new chess set)
python demo_chess_recognition.py --transfer white_view.jpg black_view.jpg

# Test individual components
python demo_chess_recognition.py --test-components test_image.jpg
```

## Board Localisation

This module identifies and localizes the chess board in an image by detecting its edges and grid intersection points.

### CANNY EDGE DETECTOR:

1. **Noise Reduction (Gaussian filter)**: Smooths the image to remove noise.
2. **Finding Intensity Gradient (Sobel operator)**: Calculates gradient magnitude and direction.
3. **Non-Maximum Suppression (NMS)**: Thins edges by selecting pixels with maximum gradient magnitude.
4. **Double Thresholding**: Classifies pixels as strong, weak, or non-edges.
5. **Edge Tracking by Hysteresis**: Finalizes edge detection by including weak edges connected to strong edges.

### HOUGH TRANSFORM:

- A line in the image can be represented in parameter space.
- Instead of looking at pixels in the image, we look at which parameters (θ, ρ) define possible lines passing through those pixels.
- Then, we vote in an accumulator array for the parameters.
- Peaks in the accumulator → detected lines.

### Board Detection Process:

![Chess Board Detection Process](board-localisation/results/all_together.jpg)
>>>>>>> 43d1f27 (Feat: Code for the board localisation along with the results)
