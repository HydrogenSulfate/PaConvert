#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

# Simple test for class inheritance
test_code = '''import torch.nn as nn

class ConvNet(nn.Module):
    pass
'''

print("Testing class inheritance...")
print("Input:")
print(test_code)

try:
    from paconvert.converter import Converter
    import tempfile
    
    # Create temp files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(test_code)
        input_file = f.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        output_file = f.name
    
    # Convert with logging enabled
    converter = Converter(backend="libcst", log_level="DEBUG")
    success, failed = converter.run(input_file, output_file)
    
    # Read result
    with open(output_file, 'r') as f:
        result = f.read()
    
    print("\nOutput:")
    print(result)
    print(f"\nSuccess: {success}, Failed: {failed}")
    
    # Check specific issue
    if "paddle.nn.Layer" in result:
        print("✓ Base class converted correctly")
    elif "nn.Module" in result:
        print("✗ Base class not converted")
    
    # Cleanup
    os.unlink(input_file)
    os.unlink(output_file)
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()