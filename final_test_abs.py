#!/usr/bin/env python3
"""Final test for torch.abs conversion"""

import tempfile
import os

def test_abs_conversion():
    """Test torch.abs conversion with out parameter"""
    
    test_cases = [
        ("Simple case", "torch.abs(torch.tensor([1, -2, 3]))"),
        ("With out", "torch.abs(torch.tensor([1, -2, 3]), out=out)"),
        ("With input keyword", "torch.abs(input=torch.tensor([1, -2, 3]))"),
        ("With both", "torch.abs(input=torch.tensor([1, -2, 3]), out=out)"),
        ("Reversed order", "torch.abs(out=out, input=torch.tensor([1, -2, 3]))")
    ]
    
    for case_name, expr in test_cases:
        print(f"\n=== {case_name} ===")
        
        test_code = f'''import torch
a = [1, -2, 3]
out = torch.zeros(3)
result = {expr}
'''
        
        print("Input:")
        print(expr)
        
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
            
            # Extract the result line
            lines = result.strip().split('\n')
            result_line = None
            for line in lines:
                if line.strip().startswith('result ='):
                    result_line = line.strip()
                    break
            
            print("Output:")
            print(result_line if result_line else "No result line found")
            
            # Check expectations
            if "out=" in expr:
                if "paddle.assign" in (result_line or ""):
                    print("✅ paddle.assign used for out parameter")
                else:
                    print("❌ paddle.assign not used for out parameter")
            
            if "input=" in expr:
                if "x=" in (result_line or ""):
                    print("✅ input parameter renamed to x")
                else:
                    print("❌ input parameter not renamed to x")
            
            os.unlink(input_file)
            os.unlink(output_file)
            
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    test_abs_conversion()