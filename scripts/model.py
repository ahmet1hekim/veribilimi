"""
PyTorch Neural Network Model for California Housing Price Prediction

This module defines the neural network architecture for regression task.
"""

import torch
import torch.nn as nn

class HousingPriceModel(nn.Module):
    """
    Neural Network for predicting house prices
    
    Architecture:
    - Input layer: number of features
    - Hidden layer 1: 128 neurons + ReLU + Dropout
    - Hidden layer 2: 64 neurons + ReLU + Dropout
    - Hidden layer 3: 32 neurons + ReLU + Dropout
    - Output layer: 1 neuron (price prediction)
    """
    
    def __init__(self, input_dim, dropout_rate=0.2):
        """
        Initialize the model
        
        Args:
            input_dim (int): Number of input features
            dropout_rate (float): Dropout probability for regularization
        """
        super(HousingPriceModel, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.dropout3 = nn.Dropout(dropout_rate)
        
        self.fc4 = nn.Linear(32, 1)
        
        self.relu = nn.ReLU()
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
            
        Returns:
            torch.Tensor: Output predictions of shape (batch_size, 1)
        """
        # Layer 1
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        
        # Layer 2
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        
        # Layer 3
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout3(x)
        
        # Output layer
        x = self.fc4(x)
        
        return x
