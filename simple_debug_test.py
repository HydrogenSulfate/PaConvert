#!/usr/bin/env python3
"""Simple debug test"""

import tempfile
import os

def test_simple():
    """Test simple case"""
    
    test_code = '''import torch
result = torch.abs(out=out, input=torch.tensor(a))
'''
    
    print("Testing simple case:")
    print(test_code)
    
    try:
        from paconvert.converter import Converter
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_code)
            input_file = f.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            output_file = f.name
        
        # Enable logging to see what's happening
        converter = Converter(backend="libcst", log_dir=".")
        success, failed = converter.run(input_file, output_file)
        
        with open(output_file, 'r') as f:
            result = f.read()
        
        print("Result:")
        print(result)
        print(f"Success: {success}, Failed: {failed}")
        
        # Check if log files were created
        log_files = [f for f in os.listdir('.') if f.endswith('.log')]
        if log_files:
            print(f"\nLog files created: {log_files}")
            for log_file in log_files:
                print(f"\n--- {log_file} ---")
                with open(log_file, 'r') as f:
                    print(f.read())
                os.unlink(log_file)  # Clean up
        
        os.unlink(input_file)
        os.unlink(output_file)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_simple()