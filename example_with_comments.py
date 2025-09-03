#!/usr/bin/env python3
"""
Example demonstrating native libcst backend's comment preservation capability
"""

# This is a comprehensive example showing torch to paddle conversion
import torch
import torch.nn as nn
import torch.nn.functional as F

# Define a neural network model
class ConvNet(nn.Module):
    """A convolutional neural network for image classification"""
    
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # First conv layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Second conv layer
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)  # Max pooling with 2x2 kernel
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 8 * 8, 128)  # First FC layer
        self.fc2 = nn.Linear(128, num_classes)  # Output layer
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        """Forward pass through the network"""
        # Apply first conv + ReLU + pooling
        x = self.pool(F.relu(self.conv1(x)))
        
        # Apply second conv + ReLU + pooling
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten for fully connected layers
        x = x.view(-1, 64 * 8 * 8)
        
        # First FC layer with ReLU and dropout
        x = self.dropout(F.relu(self.fc1(x)))
        
        # Output layer
        x = self.fc2(x)
        
        return x

# Training function
def train_model():
    """Train the model with sample data"""
    # Create model instance
    model = ConvNet(num_classes=10)
    
    # Create sample data
    batch_size = 32
    input_data = torch.randn(batch_size, 3, 32, 32)  # Random input images
    labels = torch.randint(0, 10, (batch_size,))  # Random labels
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # Cross entropy for classification
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer
    
    # Training step
    model.train()  # Set to training mode
    
    # Forward pass
    outputs = model(input_data)
    loss = criterion(outputs, labels)
    
    # Backward pass
    optimizer.zero_grad()  # Clear gradients
    loss.backward()  # Compute gradients
    optimizer.step()  # Update parameters
    
    print(f"Training loss: {loss.item():.4f}")  # Print loss
    
    return model

# Utility functions
def create_tensor_examples():
    """Examples of tensor creation and manipulation"""
    # Basic tensor creation
    a = torch.tensor([1, 2, 3, 4])  # Create from list
    b = torch.zeros(3, 4)  # Zero tensor
    c = torch.ones(2, 3)  # Ones tensor
    d = torch.randn(2, 3)  # Random normal tensor
    
    # Tensor operations
    result1 = torch.add(a, 5)  # Add scalar
    result2 = torch.matmul(c, d.T)  # Matrix multiplication
    result3 = torch.cat([b, c], dim=0)  # Concatenation
    
    # Tensor properties
    print(f"Shape of a: {a.shape}")  # Print shape
    print(f"Device of b: {b.device}")  # Print device
    print(f"Dtype of c: {c.dtype}")  # Print data type
    
    return a, b, c, d

# Main execution
if __name__ == "__main__":
    print("Running PyTorch example...")
    
    # Train the model
    trained_model = train_model()
    
    # Create tensor examples
    tensors = create_tensor_examples()
    
    print("Example completed successfully!")