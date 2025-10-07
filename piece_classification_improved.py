"""
Piece Classification for Chess Recognition
Based on "Determining Chess Game State From an Image" by Wölflin & Arandjelović

This implementation follows Section 4.3 of the paper:
- 12-class CNN classifier (6 piece types × 2 colors)
- InceptionV3 architecture with best performance
- Proper bounding box handling for tall pieces
- Horizontal flipping for left-side pieces
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
import os
from typing import Tuple, List, Optional, Dict
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

# Chess piece classes as per the paper
PIECE_CLASSES = [
    'white_pawn', 'white_knight', 'white_bishop', 'white_rook', 'white_queen', 'white_king',
    'black_pawn', 'black_knight', 'black_bishop', 'black_rook', 'black_queen', 'black_king'
]

class ChessPieceDataset(Dataset):
    """Dataset for chess piece classification with proper bounding box handling"""
    
    def __init__(self, pieces: List[np.ndarray], labels: List[int], transform=None):
        self.pieces = pieces
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.pieces)
    
    def __getitem__(self, idx):
        piece = self.pieces[idx]
        label = self.labels[idx]
        
        # Convert to PIL Image for transforms
        if isinstance(piece, np.ndarray):
            if len(piece.shape) == 3 and piece.shape[2] == 3:
                piece = Image.fromarray(cv2.cvtColor(piece, cv2.COLOR_BGR2RGB))
            else:
                piece = Image.fromarray(piece)
        
        if self.transform:
            piece = self.transform(piece)
        
        return piece, label

class PieceClassifier(nn.Module):
    """InceptionV3-based classifier for chess piece recognition"""
    
    def __init__(self, num_classes=12, pretrained=True):
        super(PieceClassifier, self).__init__()
        
        # Use InceptionV3 as the base model (best performance according to paper)
        self.backbone = models.inception_v3(pretrained=pretrained, aux_logits=False)
        
        # Modify the final layer for 12-class classification
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        
        # Add dropout for regularization
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # InceptionV3 requires input size of 299x299
        if x.size(-1) != 299:
            x = nn.functional.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        
        return self.backbone(x)

class ChessPieceSystem:
    """Complete piece classification system"""
    
    def __init__(self, model_path: str = None, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = PieceClassifier(num_classes=12, pretrained=True).to(self.device)
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        
        # Define transforms as per paper specifications
        self.train_transform = transforms.Compose([
            transforms.Resize((299, 299)),  # InceptionV3 requires 299x299
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.RandomAffine(degrees=0, shear=10),  # Shearing as mentioned in paper
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def extract_piece_bounding_boxes(self, warped_board: np.ndarray, 
                                   occupancy_mask: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int]]]:
        """
        Extract pieces with proper bounding boxes for tall pieces as described in paper
        
        Args:
            warped_board: Rectified chessboard image
            occupancy_mask: 8x8 binary mask indicating occupied squares
        
        Returns:
            List of (piece_image, (row, col)) tuples
        """
        h, w = warped_board.shape[:2]
        square_h = h // 8
        square_w = w // 8
        
        pieces = []
        
        for row in range(8):
            for col in range(8):
                if occupancy_mask[row, col] == 1:  # Only extract occupied squares
                    # Base square coordinates
                    y1 = row * square_h
                    y2 = (row + 1) * square_h
                    x1 = col * square_w
                    x2 = (col + 1) * square_w
                    
                    # Extend bounding box for tall pieces (kings, queens)
                    # Height extension based on position (further back = taller extension)
                    height_extension = self._calculate_height_extension(row, col)
                    width_extension = self._calculate_width_extension(col)
                    
                    # Apply extensions
                    y1_ext = max(0, y1 - height_extension)
                    y2_ext = min(h, y2 + height_extension//2)  # Less extension downward
                    x1_ext = max(0, x1 - width_extension)
                    x2_ext = min(w, x2 + width_extension)
                    
                    # Extract piece with extended bounding box
                    piece_img = warped_board[y1_ext:y2_ext, x1_ext:x2_ext]
                    
                    # Horizontal flip for left side pieces (as mentioned in paper)
                    if col < 4:  # Left side of board
                        piece_img = cv2.flip(piece_img, 1)
                    
                    # Ensure the target square is in bottom-left as per paper
                    piece_img = self._ensure_target_square_position(piece_img, row, col)
                    
                    if piece_img.size > 0:
                        pieces.append((piece_img, (row, col)))
        
        return pieces
    
    def _calculate_height_extension(self, row: int, col: int) -> int:
        """Calculate height extension based on piece position (further back = more extension)"""
        # Pieces further back (lower row numbers) need more height extension
        # to capture tall pieces like kings and queens
        base_extension = 20
        row_factor = (7 - row) * 5  # More extension for back ranks
        return base_extension + row_factor
    
    def _calculate_width_extension(self, col: int) -> int:
        """Calculate width extension based on horizontal position"""
        # Small width extension for context
        return 10
    
    def _ensure_target_square_position(self, piece_img: np.ndarray, row: int, col: int) -> np.ndarray:
        """Ensure the target square is in the bottom-left of the image as per paper"""
        # This is a simplified implementation
        # In practice, you would need more sophisticated positioning logic
        return piece_img
    
    def predict_pieces(self, pieces: List[np.ndarray]) -> Tuple[List[int], List[float], List[str]]:
        """
        Predict piece types for a list of piece images
        
        Returns:
            predictions: List of class indices
            confidences: List of confidence scores  
            class_names: List of predicted class names
        """
        if not pieces:
            return [], [], []
        
        # Prepare dataset
        dataset = ChessPieceDataset(pieces, [0] * len(pieces), self.val_transform)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
        
        predictions = []
        confidences = []
        class_names = []
        
        self.model.eval()
        with torch.no_grad():
            for batch_pieces, _ in dataloader:
                batch_pieces = batch_pieces.to(self.device)
                outputs = self.model(batch_pieces)
                
                # Apply softmax to get probabilities
                probs = torch.softmax(outputs, dim=1)
                
                # Get predictions and confidence scores
                batch_confidences, batch_preds = torch.max(probs, 1)
                
                predictions.extend(batch_preds.cpu().numpy().tolist())
                confidences.extend(batch_confidences.cpu().numpy().tolist())
                
                # Convert predictions to class names
                batch_class_names = [PIECE_CLASSES[pred] for pred in batch_preds.cpu().numpy().tolist()]
                class_names.extend(batch_class_names)
        
        return predictions, confidences, class_names
    
    def train_model(self, train_pieces: List[np.ndarray], train_labels: List[int],
                   val_pieces: List[np.ndarray], val_labels: List[int],
                   epochs: int = 100, lr: float = 0.001, batch_size: int = 16,
                   save_path: str = 'piece_classifier_model.pth'):
        """Train the piece classification model with two-stage approach as per paper"""
        
        # Create datasets
        train_dataset = ChessPieceDataset(train_pieces, train_labels, self.train_transform)
        val_dataset = ChessPieceDataset(val_pieces, val_labels, self.val_transform)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Two-stage training as described in paper
        # Stage 1: Train only classification head
        print("Stage 1: Training classification head only...")
        self._train_stage(train_loader, val_loader, epochs//4, lr, freeze_backbone=True)
        
        # Stage 2: Train entire network with lower learning rate
        print("Stage 2: Fine-tuning entire network...")
        self._train_stage(train_loader, val_loader, epochs, lr/10, freeze_backbone=False)
        
        # Save final model
        self.save_model(save_path)
    
    def _train_stage(self, train_loader, val_loader, epochs, lr, freeze_backbone=False):
        """Train model for one stage"""
        
        # Freeze/unfreeze backbone
        for param in self.model.backbone.parameters():
            param.requires_grad = not freeze_backbone
        
        # Always train the final classification layer
        for param in self.model.backbone.fc.parameters():
            param.requires_grad = True
        
        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        
        if freeze_backbone:
            # Only optimize classification head parameters
            optimizer = optim.Adam(self.model.backbone.fc.parameters(), lr=lr, weight_decay=1e-4)
        else:
            # Optimize all parameters
            optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # Training history
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_pieces, batch_labels in train_loader:
                batch_pieces = batch_pieces.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_pieces)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_labels.size(0)
                train_correct += (predicted == batch_labels).sum().item()
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_pieces, batch_labels in val_loader:
                    batch_pieces = batch_pieces.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                    
                    outputs = self.model(batch_pieces)
                    loss = criterion(outputs, batch_labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += batch_labels.size(0)
                    val_correct += (predicted == batch_labels).sum().item()
            
            # Calculate metrics
            train_loss_avg = train_loss / len(train_loader)
            val_loss_avg = val_loss / len(val_loader)
            train_acc = train_correct / train_total
            val_acc = val_correct / val_total
            
            train_losses.append(train_loss_avg)
            val_losses.append(val_loss_avg)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)
            
            scheduler.step(val_loss_avg)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
            
            if epoch % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}] - '
                      f'Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc:.4f}, '
                      f'Val Loss: {val_loss_avg:.4f}, Val Acc: {val_acc:.4f}')
        
        print(f'Stage completed. Best validation accuracy: {best_val_acc:.4f}')
    
    def evaluate_model(self, test_pieces: List[np.ndarray], test_labels: List[int]):
        """Evaluate the model on test data"""
        predictions, confidences, class_names = self.predict_pieces(test_pieces)
        
        # Calculate metrics
        accuracy = accuracy_score(test_labels, predictions)
        cm = confusion_matrix(test_labels, predictions)
        
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test Samples: {len(test_pieces)}")
        
        # Detailed classification report
        print("\nClassification Report:")
        print(classification_report(test_labels, predictions, target_names=PIECE_CLASSES))
        
        # Plot confusion matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=PIECE_CLASSES, yticklabels=PIECE_CLASSES)
        plt.title('Piece Classification Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('piece_classification_confusion_matrix.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return accuracy, cm, predictions, confidences
    
    def visualize_predictions(self, pieces: List[np.ndarray], predictions: List[int], 
                            confidences: List[float], num_samples: int = 16):
        """Visualize model predictions on sample pieces"""
        indices = np.random.choice(len(pieces), min(num_samples, len(pieces)), replace=False)
        
        fig, axes = plt.subplots(4, 4, figsize=(16, 16))
        axes = axes.ravel()
        
        for i, idx in enumerate(indices):
            piece = pieces[idx]
            pred = predictions[idx]
            conf = confidences[idx]
            
            if len(piece.shape) == 3:
                axes[i].imshow(cv2.cvtColor(piece, cv2.COLOR_BGR2RGB))
            else:
                axes[i].imshow(piece, cmap='gray')
            
            axes[i].set_title(f'{PIECE_CLASSES[pred]}\nConf: {conf:.3f}', fontsize=10)
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig('piece_classification_predictions_sample.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def save_model(self, path: str):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_architecture': 'inception_v3',
            'num_classes': 12,
            'class_names': PIECE_CLASSES
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load a trained model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {path}")

def create_sample_piece_dataset():
    """Create sample dataset for testing (placeholder function)"""
    pieces = []
    labels = []
    
    # Create dummy pieces for each class for testing
    for class_idx in range(12):
        for _ in range(10):  # 10 samples per class
            # Random piece image (this would be real chess piece images in practice)
            piece = np.random.randint(0, 256, (150, 100, 3), dtype=np.uint8)
            pieces.append(piece)
            labels.append(class_idx)
    
    return pieces, labels

def test_piece_system():
    """Test the piece classification system"""
    print("Testing Piece Classification System")
    
    # Create sample data
    pieces, labels = create_sample_piece_dataset()
    
    # Split data
    split_idx = int(0.8 * len(pieces))
    train_pieces = pieces[:split_idx]
    train_labels = labels[:split_idx]
    test_pieces = pieces[split_idx:]
    test_labels = labels[split_idx:]
    
    # Initialize system
    system = ChessPieceSystem()
    
    # Test predictions without training (using pretrained features)
    print("\nTesting predictions...")
    predictions, confidences, class_names = system.predict_pieces(test_pieces[:5])
    
    print(f"Made predictions for {len(predictions)} pieces")
    print(f"Sample predictions: {class_names}")
    print(f"Sample confidences: {[f'{c:.3f}' for c in confidences]}")

if __name__ == "__main__":
    test_piece_system()
