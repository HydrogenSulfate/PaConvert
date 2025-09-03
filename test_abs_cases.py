#!/usr/bin/env python3
"""Test the specific failing torch.abs cases"""

import tempfile
import os

def test_case(case_name, code):
    """Test a specific case"""
    print(f"\n=== {case_name} ===")
    print("Input:")
    print(code)
    
    try:
        from paconvert.converter import Converter
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            input_file = f.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            output_file = f.name
        
        converter = Converter(backend="libcst", log_dir="disable")
        success, failed = converter.run(input_file, output_file)
        
        with open(output_file, 'r') as f:
            result = f.read()
        
        print("Output:")
        print(result)
        print(f"Success: {success}, Failed: {failed}")
        
        # Check for common issues
        issues = []
        if "torch.abs" in result:
            issues.append("torch.abs not converted")
        if "out=" in result and "paddle.assign" not in result:
            issues.append("out parameter not handled with paddle.assign")
        if "input=" in result:
            issues.append("input parameter not renamed to x")
        
        if issues:
            print("Issues found:")
            for issue in issues:
                print(f"  ❌ {issue}")
            return False
        else:
            print("✅ Conversion looks good")
            return True
        
        os.unlink(input_file)
        os.unlink(output_file)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Test the failing cases"""
    
    # Test case 3: torch.abs(torch.tensor(a), out=out)
    case3 = '''import torch
a = [-1, -2, 3]
out = torch.tensor(a)
result = torch.abs(torch.tensor(a), out=out)
'''
    
    # Test case 6: torch.abs(input=torch.tensor(a), out=out)
    case6 = '''import torch
a = [-1, -2, 3]
out = torch.tensor(a)
result = torch.abs(input=torch.tensor(a), out=out)
'''
    
    # Test case 7: torch.abs(out=out, input=torch.tensor(a))
    case7 = '''import torch
a = [-1, -2, 3]
out = torch.tensor(a)
result = torch.abs(out=out, input=torch.tensor(a))
'''
    
    results = []
    results.append(test_case("Case 3 - Positional + out", case3))
    results.append(test_case("Case 6 - input keyword + out", case6))
    results.append(test_case("Case 7 - out + input keyword", case7))
    
    print(f"\n=== Summary ===")
    for i, result in enumerate(results, 3):
        status = "✅" if result else "❌"
        print(f"Case {i}: {status}")
    
    if all(results):
        print("\n🎉 All test cases passed!")
    else:
        print("\n❌ Some test cases failed")

if __name__ == "__main__":
    main()