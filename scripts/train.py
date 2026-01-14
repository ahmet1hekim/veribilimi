"""
Training Script for California Housing Price Prediction Model

This script trains a PyTorch neural network on the preprocessed data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
from model import HousingPriceModel

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class HousingDataset(Dataset):
    """Custom Dataset for California Housing data"""
    
    def __init__(self, X, y):
        """
        Args:
            X (pd.DataFrame or np.ndarray): Features
            y (pd.Series or np.ndarray): Target values
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = y.values
        
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).reshape(-1, 1)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_data(data_dir='data/cleaned'):
    """Load preprocessed data"""
    print("Loading preprocessed data...")
    
    X_train = pd.read_csv(f'{data_dir}/X_train.csv')
    X_test = pd.read_csv(f'{data_dir}/X_test.csv')
    y_train = pd.read_csv(f'{data_dir}/y_train.csv')
    y_test = pd.read_csv(f'{data_dir}/y_test.csv')
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}\n")
    
    return X_train, X_test, y_train, y_test

def create_dataloaders(X_train, X_test, y_train, y_test, batch_size=64):
    """Create PyTorch DataLoaders"""
    train_dataset = HousingDataset(X_train, y_train)
    test_dataset = HousingDataset(X_test, y_test)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    print(f"Created DataLoaders with batch size: {batch_size}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}\n")
    
    return train_loader, test_loader

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss

def evaluate(model, test_loader, criterion, device):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(test_loader)
    return avg_loss

def calculate_metrics(model, test_loader, device):
    """Calculate additional metrics (MAE, RMSE, R²)"""
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            output = model(data)
            predictions.extend(output.cpu().numpy())
            actuals.extend(target.numpy())
    
    predictions = np.array(predictions).flatten()
    actuals = np.array(actuals).flatten()
    
    # Calculate metrics
    mae = np.mean(np.abs(predictions - actuals))
    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
    
    # R² Score
    ss_res = np.sum((actuals - predictions) ** 2)
    ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    return mae, rmse, r2, predictions, actuals

def plot_training_history(history, save_path='weights/training_history.png'):
    """Plot training and validation loss"""
    plt.figure(figsize=(12, 5))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss', linewidth=2)
    plt.plot(history['val_loss'], label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Learning rate plot (if using scheduler)
    if 'learning_rate' in history:
        plt.subplot(1, 2, 2)
        plt.plot(history['learning_rate'], linewidth=2, color='green')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Learning Rate', fontsize=12)
        plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved training history plot to {save_path}")
    plt.close()

def plot_predictions(actuals, predictions, save_path='weights/predictions.png'):
    """Plot actual vs predicted values"""
    plt.figure(figsize=(10, 10))
    
    plt.scatter(actuals, predictions, alpha=0.5, s=10)
    plt.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 
             'r--', lw=2, label='Perfect Prediction')
    
    plt.xlabel('Actual Price ($)', fontsize=12)
    plt.ylabel('Predicted Price ($)', fontsize=12)
    plt.title('Actual vs Predicted House Prices', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved predictions plot to {save_path}")
    plt.close()

def train_model(
    model,
    train_loader,
    test_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    epochs=100,
    early_stopping_patience=15
):
    """Main training loop with early stopping"""
    print("="*60)
    print("STARTING TRAINING")
    print("="*60 + "\n")
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rate': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Evaluate
        val_loss = evaluate(model, test_loader, criterion, device)
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            history['learning_rate'].append(current_lr)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}]")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            if scheduler is not None:
                print(f"  LR:         {current_lr:.6f}")
            print()
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, 'weights/best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                print(f"Best validation loss: {best_val_loss:.4f}")
                break
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60 + "\n")
    
    return history

def main():
    """Main training pipeline"""
    print("\n" + "="*60)
    print("CALIFORNIA HOUSING PRICE PREDICTION - MODEL TRAINING")
    print("="*60 + "\n")
    
    # Create weights directory
    Path('weights').mkdir(exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load data
    X_train, X_test, y_train, y_test = load_data()
    
    # Create dataloaders
    train_loader, test_loader = create_dataloaders(
        X_train, X_test, y_train, y_test, batch_size=64
    )
    
    # Initialize model
    input_dim = X_train.shape[1]
    model = HousingPriceModel(input_dim=input_dim, dropout_rate=0.2).to(device)
    
    print(f"Model Architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Train model
    history = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=100,
        early_stopping_patience=15
    )
    
    # Load best model
    checkpoint = torch.load('weights/best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch']+1}")
    print(f"Best validation loss: {checkpoint['val_loss']:.4f}\n")
    
    # Calculate final metrics
    print("="*60)
    print("FINAL EVALUATION")
    print("="*60 + "\n")
    
    mae, rmse, r2, predictions, actuals = calculate_metrics(model, test_loader, device)
    
    print(f"Test Set Metrics:")
    print(f"  MAE (Mean Absolute Error):  ${mae:,.2f}")
    print(f"  RMSE (Root Mean Squared Error): ${rmse:,.2f}")
    print(f"  R² Score: {r2:.4f}")
    
    # Save metrics
    metrics = {
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
        'best_epoch': int(checkpoint['epoch']),
        'best_val_loss': float(checkpoint['val_loss'])
    }
    
    with open('weights/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"\nSaved metrics to weights/metrics.json")
    
    # Save training history
    with open('weights/training_history.json', 'w') as f:
        json.dump(history, f, indent=4)
    print(f"Saved training history to weights/training_history.json")
    
    # Plot results
    plot_training_history(history)
    plot_predictions(actuals, predictions)
    
    print("\n" + "="*60)
    print("ALL DONE!")
    print("="*60)
    print("\nModel and results saved in 'weights/' directory\n")

if __name__ == "__main__":
    main()
