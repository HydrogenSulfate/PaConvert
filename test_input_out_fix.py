#!/usr/bin/env python3
"""Test the input and out parameter fixes"""

import tempfile
import os

def test_conversion(case_name, torch_expr, expected_paddle):
    """Test a specific conversion case"""
    print(f"\n=== {case_name} ===")
    
    test_code = f'''import torch
a = [1, -2, 3]
out = torch.zeros(3)
result = {torch_expr}
'''
    
    print(f"Input:    {torch_expr}")
    print(f"Expected: {expected_paddle}")
    
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
        
        print(f"Actual:   {result_line if result_line else 'No result line found'}")
        
        # Check if it matches expected
        if result_line and expected_paddle in result_line:
            print("✅ PASS")
            return True
        else:
            print("❌ FAIL")
            return False
        
        os.unlink(input_file)
        os.unlink(output_file)
        
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    """Test all cases"""
    
    test_cases = [
        # Case 1: Simple positional argument
        ("Simple positional", 
         "torch.abs(torch.tensor(a))", 
         "paddle.abs(paddle.to_tensor(a))"),
        
        # Case 2: input keyword should become positional
        ("input keyword → positional", 
         "torch.abs(input=torch.tensor(a))", 
         "paddle.abs(paddle.to_tensor(a))"),
        
        # Case 3: out parameter should use paddle.assign
        ("out parameter", 
         "torch.abs(torch.tensor(a), out=out)", 
         "paddle.assign(paddle.abs(paddle.to_tensor(a)), output=out)"),
        
        # Case 4: input keyword + out parameter
        ("input keyword + out", 
         "torch.abs(input=torch.tensor(a), out=out)", 
         "paddle.assign(paddle.abs(paddle.to_tensor(a)), output=out)"),
        
        # Case 5: out + input (reversed order)
        ("out + input (reversed)", 
         "torch.abs(out=out, input=torch.tensor(a))", 
         "paddle.assign(paddle.abs(paddle.to_tensor(a)), output=out)"),
    ]
    
    results = []
    for case_name, torch_expr, expected_paddle in test_cases:
        result = test_conversion(case_name, torch_expr, expected_paddle)
        results.append(result)
    
    print(f"\n=== Summary ===")
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("❌ Some tests failed")

if __name__ == "__main__":
    main()