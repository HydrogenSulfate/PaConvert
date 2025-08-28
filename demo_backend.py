#!/usr/bin/env python3
"""
Demo script showing the new backend functionality.
"""

import sys
import os
import tempfile

# Add paconvert to path
sys.path.insert(0, os.path.dirname(__file__))

def demo_backend_selection():
    """Demonstrate backend selection functionality."""
    print("=== PaConvert Backend Integration Demo ===\n")
    
    # Import our backend system
    from paconvert.backend.manager import BackendManager
    
    print("1. Available backends:")
    backends = BackendManager.get_valid_backends()
    for backend in backends:
        print(f"   - {backend}")
    
    print("\n2. Testing astor backend:")
    try:
        manager = BackendManager("astor")
        print(f"   ✓ Created backend: {manager.get_backend_name()}")
        
        # Test with sample PyTorch code
        sample_code = """import torch
import torch.nn as nn

# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)  # Linear layer
    
    def forward(self, x):
        return torch.relu(self.linear(x))  # Apply ReLU
"""
        
        print("   ✓ Parsing sample PyTorch code...")
        tree = manager.parse_code(sample_code)
        
        print("   ✓ Generating code...")
        generated = manager.generate_code(tree)
        
        print("   ✓ Sample code processed successfully")
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print("\n3. Testing libcst backend:")
    try:
        manager = BackendManager("libcst")
        print(f"   ✓ Created backend: {manager.get_backend_name()}")
        
        if manager.is_fallback_used():
            print(f"   ⚠ Fallback used: requested libcst, got {manager.get_backend_name()}")
        
        # Test comment preservation
        code_with_comments = """# This is a header comment
import torch  # PyTorch import

# Model definition
x = torch.tensor([1, 2, 3])  # Create a tensor
"""
        
        print("   ✓ Parsing code with comments...")
        tree = manager.parse_code(code_with_comments)
        
        print("   ✓ Generating code...")
        generated = manager.generate_code(tree)
        
        if "# This is a header comment" in generated:
            print("   ✓ Comments preserved!")
        else:
            print("   ⚠ Comments not preserved (expected with bridge implementation)")
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print("\n4. Testing Converter integration:")
    try:
        from paconvert.converter import Converter
        
        # Create converter with astor backend
        converter = Converter(log_dir="disable", backend="astor")
        print(f"   ✓ Created Converter with {converter.backend_type} backend")
        print(f"   ✓ Active backend: {converter.backend_manager.get_backend_name()}")
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print("\n5. Command line usage examples:")
    print("   # Use default astor backend:")
    print("   paconvert -i torch_code/ -o paddle_code/")
    print("")
    print("   # Explicitly use astor backend:")
    print("   paconvert -i torch_code/ -o paddle_code/ --backend astor")
    print("")
    print("   # Use libcst backend for comment preservation:")
    print("   paconvert -i torch_code/ -o paddle_code/ --backend libcst")
    
    print("\n=== Demo Complete ===")

if __name__ == "__main__":
    demo_backend_selection()