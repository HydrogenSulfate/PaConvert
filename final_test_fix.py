#!/usr/bin/env python3

import sys
import os
import tempfile
sys.path.insert(0, os.path.dirname(__file__))

def test_final_fix():
    """Test the final fix"""
    print("Testing final fix...")
    
    test_code = '''import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.linear = nn.Linear(10, 1)
'''
    
    print("Input:")
    print(test_code)
    print("="*50)
    
    try:
        from paconvert.converter import Converter
        
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
        
        print("Output:")
        print(result)
        print("="*50)
        print(f"Success: {success}, Failed: {failed}")
        
        # Check results
        checks = [
            ("paddle import added", "import paddle" in result),
            ("no torch imports", "import torch" not in result),
            ("base class converted", "paddle.nn.Layer" in result),
            ("method calls converted", "paddle.nn.Linear" in result),
            ("no unsupported markers", ">>>>>>" not in result),
        ]
        
        all_passed = True
        for check_name, check_result in checks:
            status = "✓" if check_result else "✗"
            print(f"{status} {check_name}")
            if not check_result:
                all_passed = False
        
        # Cleanup
        os.unlink(input_file)
        os.unlink(output_file)
        
        return all_passed
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_final_fix()
    if success:
        print("\n🎉 Final fix test passed!")
    else:
        print("\n❌ Final fix test failed")