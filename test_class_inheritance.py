#!/usr/bin/env python3
"""
Test class inheritance conversion
"""

import sys
import os
import tempfile

# Add paconvert to path
sys.path.insert(0, os.path.dirname(__file__))

def test_class_inheritance():
    """Test class inheritance conversion"""
    print("Testing class inheritance conversion...")
    
    test_code = '''import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(10, 1)
'''
    
    print("Input code:")
    print(test_code)
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
            print(result)
            print(f"\nSuccess: {success_count}, Failed: {failed_count}")
            
            # Check results
            if "paddle.nn.Layer" in result:
                print("✓ Base class converted successfully")
            else:
                print("✗ Base class not converted")
            
            if "paddle.nn.Linear" in result:
                print("✓ Method calls converted successfully")
            else:
                print("✗ Method calls not converted")
            
            if ">>>>>>" not in result:
                print("✓ No unsupported API markers")
                return True
            else:
                print("✗ Still has unsupported API markers")
                return False
                
        finally:
            os.unlink(input_file)
            os.unlink(output_file)
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_class_inheritance()
    if success:
        print("\n🎉 Class inheritance test passed!")
    else:
        print("\n❌ Class inheritance test failed")