#!/usr/bin/env python3
"""
Test script to verify import handling is fixed
"""

import sys
import os
import tempfile

# Add paconvert to path
sys.path.insert(0, os.path.dirname(__file__))

def test_import_handling():
    """Test import handling with aliases"""
    print("Testing import handling with aliases...")
    
    # Test code with import aliases
    test_code = '''import torch
import torch.nn as nn
import torch.nn.functional as F

# Create a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(10, 1)
    
    def forward(self, x):
        return F.relu(self.linear(x))

# Create tensor
x = torch.tensor([1, 2, 3])
'''
    
    print("Input code:")
    print(test_code)
    print("="*50)
    
    try:
        from paconvert.backend.libcst_backend import LibcstBackend
        from paconvert.transformer.libcst_transformers.import_transformer import LibcstImportTransformer
        from paconvert.transformer.libcst_transformers.basic_transformer import LibcstBasicTransformer
        
        # Step 1: Parse code
        backend = LibcstBackend()
        tree = backend.parse_code(test_code)
        print("✓ Code parsed")
        
        # Step 2: Set up imports map
        imports_map = {}
        
        # Step 3: Apply import transformer
        print("\nApplying import transformer...")
        import_transformer = LibcstImportTransformer(
            tree, "test.py", imports_map, None, {}, {}
        )
        tree = import_transformer.transform()
        
        print(f"Imports map after import transformer: {imports_map}")
        
        # Step 4: Apply basic transformer
        print("\nApplying basic transformer...")
        basic_transformer = LibcstBasicTransformer(
            tree, "test.py", imports_map, None, {}, {}
        )
        tree = basic_transformer.transform()
        
        print(f"Torch API count: {basic_transformer.torch_api_count}")
        print(f"Success API count: {basic_transformer.success_api_count}")
        
        # Step 5: Generate final code
        final_code = backend.generate_code(tree)
        print("\n" + "="*50)
        print("Output code:")
        print(final_code)
        
        # Check results
        if "import paddle" in final_code:
            print("\n✓ Paddle import added")
        else:
            print("\n✗ Paddle import missing")
        
        if "import torch" not in final_code:
            print("✓ Torch imports removed")
        else:
            print("✗ Torch imports still present")
        
        if "paddle.nn.Linear" in final_code:
            print("✓ API conversion working")
        else:
            print("✗ API conversion not working")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_full_converter():
    """Test using the full converter"""
    print("\n" + "="*60)
    print("Testing full converter with import aliases...")
    
    test_code = '''import torch
import torch.nn as nn

x = torch.tensor([1, 2, 3])
linear = nn.Linear(10, 1)
'''
    
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
            
            print("Full converter result:")
            print(result)
            print(f"\nSuccess: {success_count}, Failed: {failed_count}")
            
            # Check if conversion worked
            has_paddle = "paddle" in result
            no_torch_imports = "import torch" not in result
            
            if has_paddle and no_torch_imports:
                print("✓ Full converter works correctly!")
                return True
            else:
                print("✗ Full converter has issues")
                return False
                
        finally:
            os.unlink(input_file)
            os.unlink(output_file)
            
    except Exception as e:
        print(f"✗ Full converter error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test1 = test_import_handling()
    test2 = test_full_converter()
    
    if test1 and test2:
        print("\n🎉 Import handling tests passed!")
    else:
        print("\n❌ Some tests failed")