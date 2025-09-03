#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

# Simple test
test_code = '''import torch
import torch.nn as nn

x = torch.tensor([1, 2, 3])
linear = nn.Linear(10, 1)
'''

try:
    from paconvert.converter import Converter
    import tempfile
    
    # Create temp files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(test_code)
        input_file = f.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        output_file = f.name
    
    # Convert
    converter = Converter(backend="libcst", log_dir="disable")
    success, failed = converter.run(input_file, output_file)
    
    # Read result
    with open(output_file, 'r') as f:
        result = f.read()
    
    print("Input:")
    print(test_code)
    print("\nOutput:")
    print(result)
    print(f"\nSuccess: {success}, Failed: {failed}")
    
    # Cleanup
    os.unlink(input_file)
    os.unlink(output_file)
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()