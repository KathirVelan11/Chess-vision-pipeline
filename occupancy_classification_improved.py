"""
Occupancy Classification for Chess Squares
Based on "Determining Chess Game State From an Image" by Wölflin & Arandjelović

This implementation follows Section 4.2 of the paper:
- Binary CNN classifier (occupied vs empty)
- ResNet architecture with best performance
- 50% context inclusion in square cropping
- Proper data augmentation and training
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
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

class ChessSquareDataset(Dataset):
    """Dataset for chess square occupancy classification with 50% context"""
    
    def __init__(self, squares: List[np.ndarray], labels: List[int], transform=None):
        self.squares = squares
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.squares)
    
    def __getitem__(self, idx):
        square = self.squares[idx]
        label = self.labels[idx]
        
        # Convert to PIL Image for transforms
        if isinstance(square, np.ndarray):
            square = Image.fromarray(square)
        
        if self.transform:
            square = self.transform(square)
        
        return square, label

class OccupancyClassifier(nn.Module):
    """ResNet-based binary classifier for chess square occupancy"""
    
    def __init__(self, pretrained=True):
        super(OccupancyClassifier, self).__init__()
        
        # Use ResNet18 as the base model (following paper's best performance)
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # Modify the final layer for binary classification
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 2)
        
        # Add dropout for regularization
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        features = self.backbone.avgpool(self.backbone.layer4(
            self.backbone.layer3(
                self.backbone.layer2(
                    self.backbone.layer1(
                        self.backbone.maxpool(
                            self.backbone.relu(
                                self.backbone.bn1(
                                    self.backbone.conv1(x)
                                )
                            )
                        )
                    )
                )
            )
        ))
        
        features = torch.flatten(features, 1)
        features = self.dropout(features)
        output = self.backbone.fc(features)
        
        return output

class ChessOccupancySystem:
    """Complete occupancy classification system"""
    
    def __init__(self, model_path: str = None, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = OccupancyClassifier(pretrained=True).to(self.device)
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        
        # Define transforms as per paper specifications
        self.train_transform = transforms.Compose([
            transforms.Resize((100, 100)),  # Paper uses 100x100 input
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize((100, 100)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def extract_squares_with_context(self, warped_board: np.ndarray, 
                                   context_factor: float = 0.5) -> List[np.ndarray]:
        """
        Extract 64 squares from warped board with 50% context as described in paper
        
        Args:
            warped_board: Rectified chessboard image
            context_factor: Factor for context inclusion (0.5 = 50% as per paper)
        
        Returns:
            List of 64 square images with context
        """
        h, w = warped_board.shape[:2]
        square_h = h // 8
        square_w = w // 8
        
        squares = []
        
        for row in range(8):
            for col in range(8):
                # Base square coordinates
                y1 = row * square_h
                y2 = (row + 1) * square_h
                x1 = col * square_w
                x2 = (col + 1) * square_w
                
                # Add context (50% increase in width and height)
                context_h = int(square_h * context_factor / 2)
                context_w = int(square_w * context_factor / 2)
                
                # Expand boundaries with context
                y1_ctx = max(0, y1 - context_h)
                y2_ctx = min(h, y2 + context_h)
                x1_ctx = max(0, x1 - context_w)
                x2_ctx = min(w, x2 + context_w)
                
                # Extract square with context
                square = warped_board[y1_ctx:y2_ctx, x1_ctx:x2_ctx]
                
                # Ensure consistent size by resizing
                if square.size > 0:
                    square = cv2.resize(square, (100, 100))  # Paper uses 100x100
                    squares.append(square)
        
        return squares
    
    def predict_occupancy(self, squares: List[np.ndarray]) -> Tuple[List[int], List[float]]:
        """
        Predict occupancy for a list of squares
        
        Returns:
            predictions: List of binary predictions (0=empty, 1=occupied)
            confidences: List of confidence scores
        """
        if not squares:
            return [], []
        
        # Prepare dataset
        dataset = ChessSquareDataset(squares, [0] * len(squares), self.val_transform)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        predictions = []
        confidences = []
        
        self.model.eval()
        with torch.no_grad():
            for batch_squares, _ in dataloader:
                batch_squares = batch_squares.to(self.device)
                outputs = self.model(batch_squares)
                
                # Apply softmax to get probabilities
                probs = torch.softmax(outputs, dim=1)
                
                # Get predictions and confidence scores
                _, batch_preds = torch.max(outputs, 1)
                batch_confidences = torch.max(probs, 1)[0]
                
                predictions.extend(batch_preds.cpu().numpy().tolist())
                confidences.extend(batch_confidences.cpu().numpy().tolist())
        
        return predictions, confidences
    
    def train_model(self, train_squares: List[np.ndarray], train_labels: List[int],
                   val_squares: List[np.ndarray], val_labels: List[int],
                   epochs: int = 50, lr: float = 0.001, batch_size: int = 32,
                   save_path: str = 'occupancy_model.pth'):
        """Train the occupancy classification model"""
        
        # Create datasets
        train_dataset = ChessSquareDataset(train_squares, train_labels, self.train_transform)
        val_dataset = ChessSquareDataset(val_squares, val_labels, self.val_transform)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # Training history
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        
        best_val_acc = 0.0
        
        print(f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples")
        print(f"Using device: {self.device}")
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_squares, batch_labels in train_loader:
                batch_squares = batch_squares.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_squares)
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
                for batch_squares, batch_labels in val_loader:
                    batch_squares = batch_squares.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                    
                    outputs = self.model(batch_squares)
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
            
            # Learning rate scheduling
            scheduler.step(val_loss_avg)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model(save_path)
            
            print(f'Epoch [{epoch+1}/{epochs}] - '
                  f'Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc:.4f}, '
                  f'Val Loss: {val_loss_avg:.4f}, Val Acc: {val_acc:.4f}')
        
        print(f'Best validation accuracy: {best_val_acc:.4f}')
        
        # Plot training curves
        self.plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies)
        
        return train_losses, val_losses, train_accuracies, val_accuracies
    
    def plot_training_curves(self, train_losses, val_losses, train_accuracies, val_accuracies):
        """Plot training and validation curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Loss curves
        ax1.plot(train_losses, label='Training Loss')
        ax1.plot(val_losses, label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy curves
        ax2.plot(train_accuracies, label='Training Accuracy')
        ax2.plot(val_accuracies, label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('occupancy_training_curves.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def evaluate_model(self, test_squares: List[np.ndarray], test_labels: List[int]):
        """Evaluate the model on test data"""
        predictions, confidences = self.predict_occupancy(test_squares)
        
        # Calculate metrics
        accuracy = accuracy_score(test_labels, predictions)
        cm = confusion_matrix(test_labels, predictions)
        
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test Samples: {len(test_squares)}")
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Empty', 'Occupied'], 
                   yticklabels=['Empty', 'Occupied'])
        plt.title('Occupancy Classification Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('occupancy_confusion_matrix.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return accuracy, cm, predictions, confidences
    
    def visualize_predictions(self, squares: List[np.ndarray], predictions: List[int], 
                            confidences: List[float], num_samples: int = 16):
        """Visualize model predictions on sample squares"""
        indices = np.random.choice(len(squares), min(num_samples, len(squares)), replace=False)
        
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        axes = axes.ravel()
        
        for i, idx in enumerate(indices):
            square = squares[idx]
            pred = predictions[idx]
            conf = confidences[idx]
            
            axes[i].imshow(square)
            axes[i].set_title(f'{"Occupied" if pred == 1 else "Empty"}\nConf: {conf:.3f}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig('occupancy_predictions_sample.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def save_model(self, path: str):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_architecture': 'resnet18'
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load a trained model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {path}")

def create_sample_dataset():
    """Create sample dataset for testing (placeholder function)"""
    # This is a placeholder - in practice, you would load real chess square images
    squares = []
    labels = []
    
    # Create some dummy squares for testing
    for i in range(100):
        # Random square (this would be real chess square images in practice)
        square = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        squares.append(square)
        labels.append(np.random.randint(0, 2))  # Random occupancy
    
    return squares, labels

def test_occupancy_system():
    """Test the occupancy classification system"""
    print("Testing Occupancy Classification System")
    
    # Create sample data
    squares, labels = create_sample_dataset()
    
    # Split data
    split_idx = int(0.8 * len(squares))
    train_squares = squares[:split_idx]
    train_labels = labels[:split_idx]
    test_squares = squares[split_idx:]
    test_labels = labels[split_idx:]
    
    # Initialize system
    system = ChessOccupancySystem()
    
    # Train model (with very few epochs for testing)
    print("\nTraining model...")
    system.train_model(train_squares, train_labels, test_squares, test_labels, epochs=2)
    
    # Test predictions
    print("\nTesting predictions...")
    predictions, confidences = system.predict_occupancy(test_squares)
    
    print(f"Made predictions for {len(predictions)} squares")
    print(f"Sample predictions: {predictions[:10]}")
    print(f"Sample confidences: {[f'{c:.3f}' for c in confidences[:10]]}")

if __name__ == "__main__":
    test_occupancy_system()
