#!/usr/bin/env python3
import sys
import os
import tempfile
sys.path.insert(0, os.path.dirname(__file__))

def test_out_parameter():
    """Test torch.abs with out parameter conversion"""
    print("Testing torch.abs with out parameter...")
    
    test_code = '''import torch
a = [-1, -2, 3]
out = torch.tensor(a)
result = torch.abs(out=out, input=torch.tensor(a))
'''
    
    print("Input code:")
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
        
        # Convert using LibCST backend
        converter = Converter(backend="libcst", log_dir="disable")
        success, failed = converter.run(input_file, output_file)
        
        # Read result
        with open(output_file, 'r') as f:
            result = f.read()
        
        print("Output code:")
        print(result)
        print("="*50)
        print(f"Success: {success}, Failed: {failed}")
        
        # Check results
        checks = [
            ("paddle import added", "import paddle" in result),
            ("no torch imports", "import torch" not in result),
            ("torch.tensor converted", "paddle.to_tensor" in result),
            ("paddle.assign used", "paddle.assign" in result),
            ("input parameter renamed to x", "x=paddle.to_tensor" in result),
            ("output parameter used", "output=out" in result),
            ("no torch.abs remaining", "torch.abs" not in result),
            ("no unsupported markers", ">>>>>>" not in result),
        ]
        
        all_passed = True
        for check_name, check_result in checks:
            status = "✓" if check_result else "✗"
            print(f"{status} {check_name}")
            if not check_result:
                all_passed = False
        
        # Check expected pattern
        expected_pattern = "paddle.assign(paddle.abs(x=paddle.to_tensor(a)), output=out)"
        pattern_found = expected_pattern in result.replace(" ", "").replace("\n", "")
        result_clean = result.replace(" ", "").replace("\n", "")
        
        print(f"\nExpected pattern: {expected_pattern}")
        print(f"Result (cleaned): {result_clean}")
        print(f"Pattern found: {'✓' if pattern_found else '✗'}")
        
        # Cleanup
        os.unlink(input_file)
        os.unlink(output_file)
        
        return all_passed and pattern_found
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_out_parameter()
    if success:
        print("\n🎉 Out parameter test passed!")
    else:
        print("\n❌ Out parameter test failed")