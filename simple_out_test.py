#!/usr/bin/env python3
"""Simple test for out parameter handling"""

import tempfile
import os

def test_simple_conversion():
    """Test simple torch.abs conversion"""
    test_code = '''import torch
result = torch.abs(torch.tensor([-1, 2, -3]))
'''
    
    print("Testing simple conversion:")
    print(test_code)
    
    try:
        from paconvert.converter import Converter
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_code)
            input_file = f.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            output_file = f.name
        
        converter = Converter(backend="libcst", log_dir="disable")
        success, failed = converter.run(input_file, output_file)
        
        with open(output_file, 'r') as f:
            result = f.read()
        
        print("Result:")
        print(result)
        print(f"Success: {success}, Failed: {failed}")
        
        os.unlink(input_file)
        os.unlink(output_file)
        
        return "paddle.abs" in result and "torch.abs" not in result
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_out_conversion():
    """Test torch.abs with out parameter"""
    test_code = '''import torch
a = torch.tensor([-1, 2, -3])
out = torch.zeros_like(a)
result = torch.abs(input=a, out=out)
'''
    
    print("\nTesting out parameter conversion:")
    print(test_code)
    
    try:
        from paconvert.converter import Converter
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_code)
            input_file = f.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            output_file = f.name
        
        converter = Converter(backend="libcst", log_dir="disable")
        success, failed = converter.run(input_file, output_file)
        
        with open(output_file, 'r') as f:
            result = f.read()
        
        print("Result:")
        print(result)
        print(f"Success: {success}, Failed: {failed}")
        
        os.unlink(input_file)
        os.unlink(output_file)
        
        return "paddle.assign" in result and "output=" in result
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== Testing LibCST Out Parameter Handling ===")
    
    simple_ok = test_simple_conversion()
    out_ok = test_out_conversion()
    
    print(f"\nResults:")
    print(f"Simple conversion: {'✓' if simple_ok else '✗'}")
    print(f"Out parameter: {'✓' if out_ok else '✗'}")
    
    if simple_ok and out_ok:
        print("\n🎉 All tests passed!")
    else:
        print("\n❌ Some tests failed")