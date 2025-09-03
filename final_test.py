#!/usr/bin/env python3
"""
Final test to verify the libcst backend works correctly
"""

import sys
import os
import tempfile

# Add paconvert to path
sys.path.insert(0, os.path.dirname(__file__))

def test_conversion():
    """Test the conversion with detailed debugging"""
    print("Testing LibCST backend conversion...")
    
    # Test code
    test_code = '''import torch
x = torch.tensor([1, 2, 3])
y = torch.zeros(3, 4)
'''
    
    print("Input code:")
    print(test_code)
    print("="*50)
    
    try:
        # Import required modules
        from paconvert.backend.libcst_backend import LibcstBackend
        from paconvert.transformer.libcst_transformers.import_transformer import LibcstImportTransformer
        from paconvert.transformer.libcst_transformers.basic_transformer import LibcstBasicTransformer
        import libcst as cst
        
        # Step 1: Create backend and parse
        backend = LibcstBackend()
        tree = backend.parse_code(test_code)
        print("✓ Code parsed successfully")
        
        # Step 2: Set up imports map
        imports_map = {}
        
        # Step 3: Apply import transformer
        print("\nApplying import transformer...")
        import_transformer = LibcstImportTransformer(
            tree, "test.py", imports_map, None, {}, {}
        )
        tree = import_transformer.transform()
        print(f"Torch API count: {import_transformer.torch_api_count}")
        print(f"Success API count: {import_transformer.success_api_count}")
        print(f"Imports map: {imports_map}")
        
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
        
        # Check if conversion worked
        if "paddle" in final_code and "torch" not in final_code:
            print("\n✓ Conversion successful!")
            return True
        elif "paddle" in final_code:
            print("\n⚠ Partial conversion (some torch references remain)")
            return True
        else:
            print("\n✗ No conversion detected")
            return False
            
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_full_converter():
    """Test using the full converter"""
    print("\n" + "="*60)
    print("Testing full converter...")
    
    test_code = '''import torch
x = torch.tensor([1, 2, 3])
y = torch.zeros(3, 4)
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
            
            if "paddle" in result:
                print("✓ Full converter works!")
                return True
            else:
                print("✗ Full converter failed")
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
    test1 = test_conversion()
    test2 = test_full_converter()
    
    if test1 and test2:
        print("\n🎉 All tests passed!")
    else:
        print("\n❌ Some tests failed")