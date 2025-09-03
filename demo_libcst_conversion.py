#!/usr/bin/env python3
"""
Demo script showing libcst backend conversion capabilities
"""

import os
import sys
import tempfile

# Add paconvert to path
sys.path.insert(0, os.path.dirname(__file__))

def create_sample_torch_code():
    """Create a sample torch code with comments for demonstration"""
    return '''#!/usr/bin/env python3
"""
Sample PyTorch code for conversion demonstration
"""

# Import PyTorch modules
import torch
import torch.nn as nn
import torch.nn.functional as F

# Define a simple neural network
class SimpleNet(nn.Module):
    """A simple feedforward neural network"""
    
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super(SimpleNet, self).__init__()
        
        # Define layers
        self.fc1 = nn.Linear(input_size, hidden_size)  # First fully connected layer
        self.fc2 = nn.Linear(hidden_size, num_classes)  # Output layer
        self.dropout = nn.Dropout(0.5)  # Dropout for regularization
    
    def forward(self, x):
        """Forward pass through the network"""
        # Flatten input
        x = x.view(x.size(0), -1)
        
        # First layer with ReLU activation
        x = F.relu(self.fc1(x))
        
        # Apply dropout
        x = self.dropout(x)
        
        # Output layer
        x = self.fc2(x)
        
        return x

def train_model():
    """Training function with sample data"""
    # Create model
    model = SimpleNet()
    
    # Create sample data
    batch_size = 32
    input_data = torch.randn(batch_size, 1, 28, 28)  # MNIST-like data
    labels = torch.randint(0, 10, (batch_size,))  # Random labels
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()  # Cross entropy loss
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
    
    print(f"Training loss: {loss.item():.4f}")
    
    return model

def tensor_operations():
    """Demonstrate various tensor operations"""
    # Basic tensor creation
    a = torch.tensor([1, 2, 3, 4])  # Create from list
    b = torch.zeros(3, 4)  # Zero tensor
    c = torch.ones(2, 3)  # Ones tensor
    d = torch.randn(2, 3)  # Random tensor
    
    # Tensor operations
    result1 = torch.add(a, 5)  # Element-wise addition
    result2 = torch.matmul(c, d.T)  # Matrix multiplication
    result3 = torch.cat([b, c], dim=0)  # Concatenation
    
    # Print tensor information
    print(f"Tensor a shape: {a.shape}")  # Shape information
    print(f"Tensor b device: {b.device}")  # Device information
    print(f"Tensor c dtype: {c.dtype}")  # Data type information
    
    return a, b, c, d

# Main execution
if __name__ == "__main__":
    print("PyTorch Demo - Ready for conversion to PaddlePaddle")
    
    # Run training demo
    trained_model = train_model()
    
    # Run tensor operations demo
    tensors = tensor_operations()
    
    print("Demo completed successfully!")
'''

def demo_conversion():
    """Demonstrate the conversion process"""
    print("LibCST Backend Conversion Demo")
    print("=" * 40)
    
    # Create sample code
    torch_code = create_sample_torch_code()
    
    print("1. Original PyTorch Code:")
    print("-" * 30)
    print(torch_code[:500] + "..." if len(torch_code) > 500 else torch_code)
    
    try:
        # Test libcst backend directly
        from paconvert.backend.libcst_backend import LibcstBackend
        
        print("\n2. Testing LibCST Backend:")
        print("-" * 30)
        
        backend = LibcstBackend()
        
        # Parse the code
        tree = backend.parse_code(torch_code)
        print("✓ Code parsed successfully with libcst")
        
        # Generate code (should be identical to input)
        generated = backend.generate_code(tree)
        print("✓ Code generated successfully")
        
        # Check comment preservation
        comments_preserved = "# Import PyTorch modules" in generated
        print(f"✓ Comments preserved: {comments_preserved}")
        
        print(f"\n3. Backend Information:")
        print(f"   Backend type: {backend.get_backend_type()}")
        print(f"   Transformers available: {len(backend.create_transformers())}")
        
        # Show transformer classes
        transformers = backend.create_transformers()
        print("   Transformer classes:")
        for transformer_class in transformers:
            print(f"     - {transformer_class.__name__}")
        
    except ImportError as e:
        print(f"✗ LibCST not available: {e}")
        print("Please install libcst: pip install libcst")
        return False
    except Exception as e:
        print(f"✗ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n4. Usage Instructions:")
    print("-" * 30)
    print("To convert PyTorch code to PaddlePaddle using libcst backend:")
    print("  paconvert -i input.py -o output.py --backend libcst")
    print("")
    print("Benefits of libcst backend:")
    print("  ✓ Preserves comments and formatting")
    print("  ✓ Type-safe CST operations")
    print("  ✓ Better code quality")
    print("  ✓ Native libcst implementation")
    
    return True

if __name__ == "__main__":
    success = demo_conversion()
    sys.exit(0 if success else 1)