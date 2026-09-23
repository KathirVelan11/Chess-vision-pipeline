"""
Inference script for chess occupancy classification.
Load trained model and classify individual images.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import cv2
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import argparse
import os
from pathlib import Path
import matplotlib.pyplot as plt

class ChessOccupancyInference:
    """
    Inference class for chess occupancy classification
    """
    
    def __init__(self, model_path: str, device: str = 'auto'):
        """
        Initialize inference class
        
        Args:
            model_path: Path to the trained model (.pth file)
            device: Device to run inference on ('cuda', 'cpu', or 'auto')
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() and device != 'cpu' 
                                 else 'cpu') if device == 'auto' else torch.device(device)
        
        # Load model
        self.model = self._load_model(model_path)
        
        # Define transforms (same as validation transforms)
        self.transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        
        # Class names
        self.class_names = {0: 'Empty', 1: 'Occupied'}
        
        print(f"Model loaded successfully on {self.device}")
    
    def _load_model(self, model_path: str) -> nn.Module:
        """Load the trained model"""
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Create model architecture (same as training)
        model = models.resnet18(weights=None)  # Don't load pretrained weights
        model.fc = nn.Linear(model.fc.in_features, 2)  # Binary classification
        
        # Load trained weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        print(f"Model loaded from epoch {checkpoint['epoch']} with validation accuracy: {checkpoint['val_acc']:.4f}")
        
        return model
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        Preprocess image for inference
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed tensor ready for model input
        """
        # Load image
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            # Assume it's already a numpy array
            image = image_path
        
        # Apply transforms
        transformed = self.transform(image=image)
        tensor_image = transformed['image']
        
        # Add batch dimension
        tensor_image = tensor_image.unsqueeze(0)
        
        return tensor_image
    
    def predict(self, image_path: str, return_confidence: bool = True) -> dict:
        """
        Predict occupancy for a single image
        
        Args:
            image_path: Path to the image file
            return_confidence: Whether to return confidence scores
            
        Returns:
            Dictionary with prediction results
        """
        # Preprocess image
        tensor_image = self.preprocess_image(image_path)
        tensor_image = tensor_image.to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(tensor_image)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
            
            predicted_class = predicted_class.item()
            confidence_score = confidence.item()
        
        result = {
            'predicted_class': predicted_class,
            'predicted_label': self.class_names[predicted_class],
            'confidence': confidence_score
        }
        
        if return_confidence:
            result['probabilities'] = {
                'Empty': probabilities[0][0].item(),
                'Occupied': probabilities[0][1].item()
            }
        
        return result
    
    def predict_batch(self, image_paths: list) -> list:
        """
        Predict occupancy for multiple images
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            List of prediction results
        """
        results = []
        for image_path in image_paths:
            try:
                result = self.predict(image_path)
                result['image_path'] = image_path
                results.append(result)
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
                results.append({
                    'image_path': image_path,
                    'error': str(e)
                })
        
        return results
    
    def visualize_prediction(self, image_path: str, save_path: str = None):
        """
        Visualize prediction with the original image
        
        Args:
            image_path: Path to the image file
            save_path: Optional path to save the visualization
        """
        # Make prediction
        result = self.predict(image_path)
        
        # Load and display image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        
        # Original image
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title('Original Image')
        plt.axis('off')
        
        # Prediction results
        plt.subplot(1, 2, 2)
        plt.text(0.1, 0.8, f"Prediction: {result['predicted_label']}", 
                fontsize=16, fontweight='bold')
        plt.text(0.1, 0.6, f"Confidence: {result['confidence']:.3f}", 
                fontsize=14)
        
        if 'probabilities' in result:
            plt.text(0.1, 0.4, "Probabilities:", fontsize=12, fontweight='bold')
            plt.text(0.1, 0.3, f"Empty: {result['probabilities']['Empty']:.3f}", fontsize=11)
            plt.text(0.1, 0.2, f"Occupied: {result['probabilities']['Occupied']:.3f}", fontsize=11)
        
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis('off')
        plt.title('Prediction Results')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        
        plt.show()


def main():
    """
    Main function - Just modify the paths below and run the script!
    """
    
    # =============================================================================
    # MODIFY THESE PATHS FOR YOUR TESTING
    # =============================================================================
    
    # Path to your trained model
    model_path = r"D:\SEM5\COMPUTER VISION\occupancy_classification\models\best_model.pth"

    # Path to the image you want to test (CHANGE THIS)
    image_path = r"C:\Users\kavee\OneDrive\Pictures\Screenshots\Screenshot 2025-10-01 081522.png"

    # Optional: Directory with multiple images for batch testing
    batch_directory = None  # Set to './test_images' if you want batch processing
    
    # Whether to show visualization
    show_visualization = True
    
    # =============================================================================
    # END OF USER CONFIGURATION
    # =============================================================================
    
    try:
        # Initialize classifier
        print("Initializing classifier...")
        classifier = ChessOccupancyInference(model_path, device='auto')
        
        if batch_directory:
            # Batch processing mode
            print(f"\n=== BATCH PROCESSING MODE ===")
            print(f"Processing all images in: {batch_directory}")
            
            # Get all image files
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            image_paths = []
            batch_path = Path(batch_directory)
            
            if not batch_path.exists():
                print(f"Error: Directory {batch_directory} does not exist!")
                return
            
            for ext in image_extensions:
                image_paths.extend(batch_path.glob(f'*{ext}'))
                image_paths.extend(batch_path.glob(f'*{ext.upper()}'))
            
            if not image_paths:
                print(f"No images found in {batch_directory}")
                return
            
            print(f"Found {len(image_paths)} images")
            
            # Process all images
            results = classifier.predict_batch([str(p) for p in image_paths])
            
            # Print results
            print(f"\n=== BATCH RESULTS ===")
            successful_predictions = [r for r in results if 'error' not in r]
            empty_count = sum(1 for r in successful_predictions if r['predicted_class'] == 0)
            occupied_count = sum(1 for r in successful_predictions if r['predicted_class'] == 1)
            
            for result in results:
                if 'error' not in result:
                    filename = Path(result['image_path']).name
                    print(f"{filename:20} -> {result['predicted_label']:8} (confidence: {result['confidence']:.3f})")
                else:
                    filename = Path(result['image_path']).name
                    print(f"{filename:20} -> ERROR: {result['error']}")
            
            print(f"\n=== SUMMARY ===")
            print(f"Successfully processed: {len(successful_predictions)}/{len(results)} images")
            print(f"Predicted Empty: {empty_count}")
            print(f"Predicted Occupied: {occupied_count}")
            
        else:
            # Single image processing mode
            print(f"\n=== SINGLE IMAGE MODE ===")
            print(f"Processing image: {image_path}")
            
            # Check if image exists
            if not Path(image_path).exists():
                print(f"Error: Image file {image_path} does not exist!")
                print("Please check the path and update the 'image_path' variable in main()")
                return
            
            # Make prediction
            result = classifier.predict(image_path)
            
            # Print results
            print(f"\n=== PREDICTION RESULTS ===")
            print(f"Image: {Path(image_path).name}")
            print(f"Prediction: {result['predicted_label']}")
            print(f"Confidence: {result['confidence']:.3f}")
            print(f"\nDetailed Probabilities:")
            print(f"  Empty: {result['probabilities']['Empty']:.3f}")
            print(f"  Occupied: {result['probabilities']['Occupied']:.3f}")
            
            # Show visualization if requested
            if show_visualization:
                print(f"\nShowing visualization...")
                classifier.visualize_prediction(image_path)
    
    except FileNotFoundError as e:
        print(f"Error: Could not find file - {e}")
        print("Please check your model path and image path")
    except Exception as e:
        print(f"Error occurred: {e}")
        print("Please check your paths and try again")


# Alternative simple functions for direct use
def test_single_image(model_path: str, image_path: str):
    """
    Simple function to test a single image
    
    Args:
        model_path: Path to your trained model
        image_path: Path to the image to test
    """
    classifier = ChessOccupancyInference(model_path)
    result = classifier.predict(image_path)
    
    print(f"Image: {image_path}")
    print(f"Prediction: {result['predicted_label']}")
    print(f"Confidence: {result['confidence']:.3f}")
    
    classifier.visualize_prediction(image_path)
    return result


def test_multiple_images(model_path: str, image_paths: list):
    """
    Simple function to test multiple images
    
    Args:
        model_path: Path to your trained model
        image_paths: List of image paths to test
    """
    classifier = ChessOccupancyInference(model_path)
    results = classifier.predict_batch(image_paths)
    
    for result in results:
        if 'error' not in result:
            print(f"{result['image_path']}: {result['predicted_label']} ({result['confidence']:.3f})")
        else:
            print(f"{result['image_path']}: ERROR - {result['error']}")
    
    return results


if __name__ == "__main__":
    main()