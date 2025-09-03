#!/usr/bin/env python3
"""
Comprehensive test for libcst backend
"""

import sys
import os
import tempfile

# Add paconvert to path
sys.path.insert(0, os.path.dirname(__file__))

def test_comprehensive_conversion():
    """Test comprehensive conversion including class inheritance"""
    print("Testing comprehensive conversion...")
    
    test_code = '''import torch
import torch.nn as nn
import torch.nn.functional as F

# Define a neural network model
class ConvNet(nn.Module):
    """A convolutional neural network for image classification"""
    
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
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

# Create model and data
model = ConvNet(num_classes=10)
x = torch.randn(5, 3, 32, 32)
output = model(x)
'''
    
    print("Input code:")
    print(test_code[:500] + "..." if len(test_code) > 500 else test_code)
    print("="*50)
    
    try:
        from paconvert.converter import Converter
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_code)
            input_file = f.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            output_file = f.name
        
        try:
            # Create converter
            converter = Converter(backend="libcst", log_dir="disable")
            
            # Run conversion
            success_count, failed_count = converter.run(input_file, output_file)
            
            # Read result
            with open(output_file, 'r') as f:
                result = f.read()
            
            print("Output code:")
            print(result[:1000] + "..." if len(result) > 1000 else result)
            print(f"\nSuccess: {success_count}, Failed: {failed_count}")
            
            # Check results
            checks = [
                ("paddle import", "import paddle" in result),
                ("no torch imports", "import torch" not in result),
                ("base class converted", "paddle.nn.Layer" in result),
                ("Conv2d converted", "paddle.nn.Conv2D" in result),
                ("Linear converted", "paddle.nn.Linear" in result),
                ("tensor creation", "paddle.randn" in result or "paddle.to_tensor" in result),
                ("no unsupported markers", ">>>>>>" not in result),
            ]
            
            all_passed = True
            for check_name, check_result in checks:
                status = "✓" if check_result else "✗"
                print(f"{status} {check_name}")
                if not check_result:
                    all_passed = False
            
            return all_passed
                
        finally:
            os.unlink(input_file)
            os.unlink(output_file)
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_comprehensive_conversion()
    if success:
        print("\n🎉 Comprehensive test passed!")
    else:
        print("\n❌ Comprehensive test failed")