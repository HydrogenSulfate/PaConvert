#!/usr/bin/env python3
"""Step by step debug of the transformation"""

import tempfile
import os

def debug_transformation():
    """Debug the transformation step by step"""
    
    test_code = '''import torch
a = [-1, -2, 3]
out = torch.tensor(a)
result = torch.abs(out=out, input=torch.tensor(a))
'''
    
    print("=== Debug Transformation ===")
    print("Input code:")
    print(test_code)
    print("="*50)
    
    try:
        # First, let's check if our transformer is being used
        from paconvert.converter import Converter
        from paconvert.global_var import GlobalManager
        
        # Check the mapping
        if "torch.abs" in GlobalManager.API_MAPPING:
            mapping = GlobalManager.API_MAPPING["torch.abs"]
            print("torch.abs mapping found:")
            print(f"  paddle_api: {mapping['paddle_api']}")
            print(f"  kwargs_change: {mapping.get('kwargs_change', {})}")
            print(f"  args_list: {mapping.get('args_list', [])}")
        else:
            print("❌ torch.abs mapping not found!")
            return
        
        # Test the conversion
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_code)
            input_file = f.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            output_file = f.name
        
        # Convert with debug logging
        converter = Converter(backend="libcst", log_dir=".")
        success, failed = converter.run(input_file, output_file)
        
        with open(output_file, 'r') as f:
            result = f.read()
        
        print("Conversion result:")
        print(result)
        print(f"Success: {success}, Failed: {failed}")
        
        # Analyze the result
        print("\n=== Analysis ===")
        if "torch.abs" in result:
            print("❌ torch.abs was not converted")
        else:
            print("✅ torch.abs was converted")
        
        if "paddle.abs" in result:
            print("✅ paddle.abs found")
            if "out=" in result:
                print("❌ out parameter still present in paddle.abs call")
            if "input=" in result:
                print("❌ input parameter not renamed to x")
            if "x=" in result:
                print("✅ input parameter renamed to x")
        
        if "paddle.assign" in result:
            print("✅ paddle.assign found")
        else:
            print("❌ paddle.assign not found")
        
        # Cleanup
        os.unlink(input_file)
        os.unlink(output_file)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_transformation()