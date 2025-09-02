#!/usr/bin/env python3
"""
Example showing the difference between astor and libcst backends
"""

# Sample PyTorch code with comments
sample_code = '''
import torch
import torch.nn as nn

# This is a simple neural network model
class SimpleModel(nn.Module):
    """A simple linear model for demonstration"""
    
    def __init__(self, input_size=10, output_size=1):
        super(SimpleModel, self).__init__()
        # Linear transformation layer
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        # Forward pass through the linear layer
        return self.linear(x)

# Create model instance
model = SimpleModel()

# Generate some random input data
x = torch.randn(5, 10)  # Batch size 5, input size 10

# Forward pass
output = model(x)
print(f"Output shape: {output.shape}")
'''

print("Original PyTorch code:")
print("=" * 50)
print(sample_code)
print("=" * 50)

print("\nThis code contains:")
print("- Import statements")
print("- Class and function definitions") 
print("- Comments (# and \"\"\")")
print("- Inline comments")
print("- Proper formatting and indentation")

print("\nWith astor backend:")
print("- Comments will be removed during conversion")
print("- Code will be reformatted according to black/isort")
print("- Functionality will be preserved")

print("\nWith libcst backend:")
print("- Comments will be preserved where possible")
print("- Original formatting will be better maintained")
print("- Same conversion accuracy as astor")

print("\nTo test both backends:")
print("1. Save this code to a file (e.g., test_model.py)")
print("2. Run: paconvert -i test_model.py -o output_astor --backend astor")
print("3. Run: paconvert -i test_model.py -o output_libcst --backend libcst")
print("4. Compare the outputs to see the difference")