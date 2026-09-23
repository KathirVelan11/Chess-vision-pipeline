"""
Few-Shot Transfer Learning for Chess Recognition
Based on "Determining Chess Game State From an Image" by Wölflin & Arandjelović

This implementation follows Section 4.4 of the paper:
- Adapt the system to new chess sets using only 2 starting position images
- Heavy data augmentation (shearing, color jittering, scaling, translation)
- Two-stage fine-tuning approach
- No manual labeling required (starting position is known)
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import our components
from board_localization_improved import ChessBoardLocalizer
from occupancy_classification_improved import ChessOccupancySystem
from piece_classification_improved import ChessPieceSystem, PIECE_CLASSES
from fen_generator import FENGenerator

class TransferLearningDataset(Dataset):
    """Dataset for few-shot transfer learning with heavy augmentation"""
    
    def __init__(self, samples: List[Tuple[np.ndarray, int]], transform=None, is_piece_dataset=False):
        self.samples = samples
        self.transform = transform
        self.is_piece_dataset = is_piece_dataset
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image, label = self.samples[idx]
        
        # Convert to PIL Image for transforms
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3:
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                image = Image.fromarray(image)
        
        # Apply augmentations
        if self.transform:
            if hasattr(self.transform, '__call__'):
                if isinstance(image, Image.Image):
                    image = np.array(image)
                transformed = self.transform(image=image)
                image = transformed['image']
            else:
                image = self.transform(image)
        
        return image, label

class ChessTransferLearningSystem:
    """System for adapting chess recognition to new chess sets"""
    
    def __init__(self, pipeline_components: Dict, device: str = 'cuda'):
        """
        Initialize transfer learning system
        
        Args:
            pipeline_components: Dictionary with 'occupancy' and 'piece' systems
            device: Computing device
        """
        self.device = device
        self.board_localizer = ChessBoardLocalizer()
        self.occupancy_system = pipeline_components['occupancy']
        self.piece_system = pipeline_components['piece'] 
        self.fen_generator = FENGenerator()
        
        # Starting position FEN
        self.starting_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        
        print("Transfer Learning System initialized")
    
    def adapt_to_new_chess_set(self, 
                              white_perspective_image: str,
                              black_perspective_image: str,
                              output_dir: str = "transfer_learning_results",
                              occupancy_epochs: int = 100,
                              piece_epochs: int = 150) -> Dict:
        """
        Adapt the system to a new chess set using two starting position images
        
        Args:
            white_perspective_image: Path to starting position from white's perspective
            black_perspective_image: Path to starting position from black's perspective
            output_dir: Directory to save results
            occupancy_epochs: Epochs for occupancy fine-tuning
            piece_epochs: Epochs for piece fine-tuning
            
        Returns:
            Dictionary with adaptation results
        """
        results = {
            'success': False,
            'occupancy_adaptation': {},
            'piece_adaptation': {},
            'validation_results': {}
        }
        
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        print("🔄 Starting Few-Shot Transfer Learning")
        print("=" * 60)
        
        try:
            # Step 1: Extract training data from starting position images
            print("Step 1: Extracting training data...")
            training_data = self._extract_training_data([white_perspective_image, black_perspective_image])
            
            if not training_data['success']:
                results['error'] = "Failed to extract training data"
                return results
            
            # Step 2: Fine-tune occupancy classifier
            print("\nStep 2: Fine-tuning occupancy classifier...")
            occupancy_results = self._fine_tune_occupancy_classifier(
                training_data['occupancy_samples'],
                training_data['occupancy_labels'],
                epochs=occupancy_epochs,
                save_path=f"{output_dir}/adapted_occupancy_model.pth"
            )
            results['occupancy_adaptation'] = occupancy_results
            
            # Step 3: Fine-tune piece classifier
            print("\nStep 3: Fine-tuning piece classifier...")
            piece_results = self._fine_tune_piece_classifier(
                training_data['piece_samples'],
                training_data['piece_labels'],
                epochs=piece_epochs,
                save_path=f"{output_dir}/adapted_piece_model.pth"
            )
            results['piece_adaptation'] = piece_results
            
            # Step 4: Validate adaptation
            print("\nStep 4: Validating adaptation...")
            validation_results = self._validate_adaptation(
                [white_perspective_image, black_perspective_image],
                output_dir
            )
            results['validation_results'] = validation_results
            
            results['success'] = True
            
            print("\n✅ Transfer learning completed successfully!")
            
        except Exception as e:
            print(f"\n❌ Transfer learning failed: {str(e)}")
            results['error'] = str(e)
        
        return results
    
    def _extract_training_data(self, image_paths: List[str]) -> Dict:
        """Extract training samples from starting position images"""
        
        occupancy_samples = []
        occupancy_labels = []
        piece_samples = []
        piece_labels = []
        
        # Parse starting position to get ground truth
        board_state, _ = self.fen_generator.fen_to_board(self.starting_fen)
        
        for image_path in image_paths:
            print(f"Processing: {image_path}")
            
            # Load and localize board
            image = cv2.imread(image_path)
            if image is None:
                continue
                
            homography, corner_points, _ = self.board_localizer.localize_board(image)
            
            if homography is None:
                print(f"⚠️  Failed to localize board in {image_path}")
                continue
            
            # Rectify board
            warped_board = cv2.warpPerspective(image, homography, (400, 400))
            
            # Extract squares for occupancy classification (64 squares per image)
            squares = self.occupancy_system.extract_squares_with_context(warped_board, context_factor=0.5)
            
            for row in range(8):
                for col in range(8):
                    square_idx = row * 8 + col
                    if square_idx < len(squares):
                        square = squares[square_idx]
                        
                        # Occupancy label (1 if occupied, 0 if empty)
                        piece = board_state[row, col]
                        is_occupied = 1 if piece != 'empty' and piece is not None else 0
                        
                        occupancy_samples.append(square)
                        occupancy_labels.append(is_occupied)
                        
                        # If occupied, add to piece classification dataset
                        if is_occupied:
                            # Extract piece with proper bounding box
                            piece_img = self._extract_piece_for_classification(warped_board, row, col)
                            
                            if piece_img is not None:
                                # Get piece class index
                                piece_class_idx = PIECE_CLASSES.index(piece) if piece in PIECE_CLASSES else 0
                                
                                piece_samples.append(piece_img)
                                piece_labels.append(piece_class_idx)
        
        print(f"Extracted {len(occupancy_samples)} occupancy samples")
        print(f"Extracted {len(piece_samples)} piece samples")
        
        return {
            'success': len(occupancy_samples) > 0 and len(piece_samples) > 0,
            'occupancy_samples': occupancy_samples,
            'occupancy_labels': occupancy_labels,
            'piece_samples': piece_samples,
            'piece_labels': piece_labels
        }
    
    def _extract_piece_for_classification(self, warped_board: np.ndarray, 
                                        row: int, col: int) -> Optional[np.ndarray]:
        """Extract piece image with proper bounding box for classification"""
        h, w = warped_board.shape[:2]
        square_h = h // 8
        square_w = w // 8
        
        # Base square coordinates
        y1 = row * square_h
        y2 = (row + 1) * square_h
        x1 = col * square_w
        x2 = (col + 1) * square_w
        
        # Extend bounding box for tall pieces
        height_extension = (7 - row) * 8 + 20  # More extension for back ranks
        width_extension = 10
        
        y1_ext = max(0, y1 - height_extension)
        y2_ext = min(h, y2 + height_extension//2)
        x1_ext = max(0, x1 - width_extension)
        x2_ext = min(w, x2 + width_extension)
        
        # Extract piece
        piece_img = warped_board[y1_ext:y2_ext, x1_ext:x2_ext]
        
        # Horizontal flip for left side pieces (as mentioned in paper)
        if col < 4:
            piece_img = cv2.flip(piece_img, 1)
        
        return piece_img if piece_img.size > 0 else None
    
    def _get_heavy_augmentation_transforms(self, is_piece_classifier: bool = False):
        """Get heavy augmentation transforms as described in the paper"""
        
        if is_piece_classifier:
            # Heavy augmentation for piece classifier as mentioned in paper
            return A.Compose([
                A.Resize(299, 299),  # InceptionV3 input size
                
                # Shearing transformation (most significant performance gain according to paper)
                A.Affine(shear=(-20, 20), p=0.8),
                
                # Color jittering
                A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2, p=0.8),
                
                # Scaling and translation
                A.Affine(scale=(0.8, 1.2), translate_percent=(-0.1, 0.1), p=0.6),
                
                # Rotation
                A.Rotate(limit=15, p=0.5),
                
                # Additional augmentations
                A.GaussNoise(var_limit=(10, 50), p=0.3),
                A.MotionBlur(blur_limit=3, p=0.2),
                A.RandomGamma(gamma_limit=(80, 120), p=0.3),
                
                # Normalize and convert to tensor
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            # Augmentation for occupancy classifier
            return A.Compose([
                A.Resize(100, 100),  # ResNet input size
                A.Rotate(limit=10, p=0.5),
                A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.6),
                A.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), p=0.4),
                A.GaussNoise(var_limit=(10, 30), p=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
    
    def _fine_tune_occupancy_classifier(self, samples: List[np.ndarray], 
                                      labels: List[int],
                                      epochs: int = 100,
                                      save_path: str = "adapted_occupancy_model.pth") -> Dict:
        """Fine-tune occupancy classifier with two-stage approach"""
        
        print(f"Fine-tuning occupancy classifier with {len(samples)} samples...")
        
        # Create augmented dataset
        augment_transform = self._get_heavy_augmentation_transforms(is_piece_classifier=False)
        
        # Create training samples with augmentation
        training_samples = []
        for sample, label in zip(samples, labels):
            training_samples.append((sample, label))
        
        dataset = TransferLearningDataset(training_samples, augment_transform)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        
        # Two-stage fine-tuning as described in paper
        model = self.occupancy_system.model
        
        # Stage 1: Train only classification head
        print("Stage 1: Training classification head...")
        self._fine_tune_stage(model, dataloader, epochs//4, lr=0.001, freeze_backbone=True)
        
        # Stage 2: Train entire network with lower learning rate
        print("Stage 2: Fine-tuning entire network...")
        self._fine_tune_stage(model, dataloader, epochs, lr=0.0001, freeze_backbone=False)
        
        # Save adapted model
        self.occupancy_system.save_model(save_path)
        
        return {
            'success': True,
            'training_samples': len(samples),
            'epochs': epochs,
            'model_path': save_path
        }
    
    def _fine_tune_piece_classifier(self, samples: List[np.ndarray], 
                                  labels: List[int],
                                  epochs: int = 150,
                                  save_path: str = "adapted_piece_model.pth") -> Dict:
        """Fine-tune piece classifier with heavy augmentation"""
        
        print(f"Fine-tuning piece classifier with {len(samples)} samples...")
        
        # Create augmented dataset with heavy augmentation
        augment_transform = self._get_heavy_augmentation_transforms(is_piece_classifier=True)
        
        training_samples = []
        for sample, label in zip(samples, labels):
            training_samples.append((sample, label))
        
        dataset = TransferLearningDataset(training_samples, augment_transform, is_piece_dataset=True)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)  # Smaller batch for InceptionV3
        
        # Two-stage fine-tuning
        model = self.piece_system.model
        
        # Stage 1: Train only classification head
        print("Stage 1: Training classification head...")
        self._fine_tune_stage(model, dataloader, epochs//3, lr=0.001, freeze_backbone=True)
        
        # Stage 2: Train entire network with lower learning rate  
        print("Stage 2: Fine-tuning entire network...")
        self._fine_tune_stage(model, dataloader, epochs, lr=0.0001, freeze_backbone=False)
        
        # Additional stage for piece classifier (as mentioned in paper)
        print("Stage 3: Additional fine-tuning...")
        self._fine_tune_stage(model, dataloader, 50, lr=0.00001, freeze_backbone=False)
        
        # Save adapted model
        self.piece_system.save_model(save_path)
        
        return {
            'success': True,
            'training_samples': len(samples),
            'epochs': epochs,
            'model_path': save_path
        }
    
    def _fine_tune_stage(self, model: nn.Module, dataloader: DataLoader, 
                        epochs: int, lr: float, freeze_backbone: bool = False):
        """Fine-tune model for one stage"""
        
        # Freeze/unfreeze backbone
        if hasattr(model, 'backbone'):
            for param in model.backbone.parameters():
                param.requires_grad = not freeze_backbone
            
            # Always train the final classification layer
            if hasattr(model.backbone, 'fc'):
                for param in model.backbone.fc.parameters():
                    param.requires_grad = True
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        
        if freeze_backbone and hasattr(model, 'backbone') and hasattr(model.backbone, 'fc'):
            optimizer = optim.Adam(model.backbone.fc.parameters(), lr=lr, weight_decay=1e-4)
        else:
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        
        model.train()
        
        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            
            for batch_data, batch_labels in dataloader:
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_data)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
            
            if epoch % 20 == 0:
                epoch_loss = running_loss / len(dataloader)
                epoch_acc = correct / total
                print(f'  Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}')
    
    def _validate_adaptation(self, test_images: List[str], output_dir: str) -> Dict:
        """Validate the adaptation by testing on the starting position images"""
        
        print("Validating adaptation on starting position images...")
        
        # Import the main pipeline
        from chess_recognition_pipeline import ChessRecognitionPipeline
        
        # Create pipeline with adapted models
        adapted_pipeline = ChessRecognitionPipeline(
            occupancy_model_path=f"{output_dir}/adapted_occupancy_model.pth",
            piece_model_path=f"{output_dir}/adapted_piece_model.pth",
            device=self.device
        )
        
        validation_results = []
        
        for i, image_path in enumerate(test_images):
            print(f"Testing on image {i+1}: {Path(image_path).name}")
            
            result = adapted_pipeline.process_image(
                image_path, 
                save_intermediate=True,
                output_dir=f"{output_dir}/validation_{i+1}"
            )
            
            # Check if FEN matches starting position
            expected_fen = self.starting_fen.split()[0]  # Just piece placement part
            actual_fen = result.get('fen', '').split()[0] if result.get('fen') else ''
            
            fen_match = expected_fen == actual_fen
            
            validation_result = {
                'image': Path(image_path).name,
                'success': result['success'],
                'fen': result.get('fen'),
                'fen_matches_expected': fen_match,
                'confidence': result.get('confidence_score', 0.0),
                'processing_time': result.get('processing_time', 0.0)
            }
            
            validation_results.append(validation_result)
            
            print(f"  Result: {'✅ SUCCESS' if result['success'] else '❌ FAILED'}")
            print(f"  FEN matches: {'✅ YES' if fen_match else '❌ NO'}")
            print(f"  Confidence: {validation_result['confidence']:.3f}")
        
        # Calculate summary statistics
        successful_adaptations = sum(1 for r in validation_results if r['success'])
        fen_matches = sum(1 for r in validation_results if r['fen_matches_expected'])
        avg_confidence = np.mean([r['confidence'] for r in validation_results if r['success']])
        
        summary = {
            'total_images': len(test_images),
            'successful_predictions': successful_adaptations,
            'fen_matches': fen_matches,
            'success_rate': successful_adaptations / len(test_images),
            'fen_accuracy': fen_matches / len(test_images),
            'average_confidence': float(avg_confidence) if successful_adaptations > 0 else 0.0,
            'detailed_results': validation_results
        }
        
        print(f"\n📊 Validation Summary:")
        print(f"  Success rate: {summary['success_rate']:.1%}")
        print(f"  FEN accuracy: {summary['fen_accuracy']:.1%}")
        print(f"  Average confidence: {summary['average_confidence']:.3f}")
        
        return summary

def demo_transfer_learning():
    """Demonstrate transfer learning adaptation"""
    print("🔄 Chess Transfer Learning Demo")
    print("=" * 60)
    
    # Initialize components (using pre-trained models or random initialization)
    from chess_recognition_pipeline import ChessRecognitionPipeline
    
    base_pipeline = ChessRecognitionPipeline()
    
    # Initialize transfer learning system
    transfer_system = ChessTransferLearningSystem({
        'occupancy': base_pipeline.occupancy_system,
        'piece': base_pipeline.piece_system
    })
    
    # For demo purposes, we'll use the same image twice
    # In practice, you'd have two different perspective images
    test_image = "/home/pranesh/chess/Computer-Vision/board-localisation/images/real_image_1.jpeg"
    
    if Path(test_image).exists():
        print(f"Demo with images: {test_image}")
        
        # Adapt to new chess set
        results = transfer_system.adapt_to_new_chess_set(
            white_perspective_image=test_image,
            black_perspective_image=test_image,  # Same image for demo
            occupancy_epochs=5,  # Reduced for demo
            piece_epochs=10      # Reduced for demo
        )
        
        if results['success']:
            print("\n✅ Transfer learning demo completed!")
            print("Check the 'transfer_learning_results' directory for outputs.")
        else:
            print(f"\n❌ Transfer learning failed: {results.get('error', 'Unknown error')}")
    else:
        print(f"\n⚠️  Demo image not found: {test_image}")

if __name__ == "__main__":
    demo_transfer_learning()
