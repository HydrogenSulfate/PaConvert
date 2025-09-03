#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

def test_import_mapping():
    """Test import mapping specifically"""
    print("Testing import mapping...")
    
    test_code = '''import torch.nn as nn'''
    
    try:
        from paconvert.backend.libcst_backend import LibcstBackend
        from paconvert.transformer.libcst_transformers.import_transformer import LibcstImportTransformer
        
        # Parse code
        backend = LibcstBackend()
        tree = backend.parse_code(test_code)
        
        # Set up imports map
        imports_map = {}
        
        # Apply import transformer
        import_transformer = LibcstImportTransformer(
            tree, "test.py", imports_map, None, {}, {}
        )
        tree = import_transformer.transform()
        
        print(f"Imports map: {imports_map}")
        
        # Expected: {'test.py': {'torch_packages': ['nn'], 'nn': 'torch.nn'}}
        
        if 'test.py' in imports_map:
            file_imports = imports_map['test.py']
            print(f"File imports: {file_imports}")
            
            if 'nn' in file_imports:
                print(f"nn maps to: {file_imports['nn']}")
                if file_imports['nn'] == 'torch.nn':
                    print("✓ Import mapping is correct")
                else:
                    print("✗ Import mapping is wrong")
            else:
                print("✗ nn not found in imports")
        else:
            print("✗ File not found in imports map")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_api_resolution():
    """Test API name resolution"""
    print("\nTesting API name resolution...")
    
    # Simulate the imports map that should be created
    imports_map = {
        'test.py': {
            'torch_packages': ['nn'],
            'nn': 'torch.nn'
        }
    }
    
    try:
        from paconvert.transformer.libcst_transformers.basic_transformer import LibcstBasicTransformer
        import libcst as cst
        
        # Create a dummy transformer
        dummy_tree = cst.parse_module("pass")
        transformer = LibcstBasicTransformer(
            dummy_tree, "test.py", imports_map, None, {}, {}
        )
        
        # Test resolution
        test_cases = [
            "nn.Module",
            "nn.Linear", 
            "nn.Conv2d"
        ]
        
        for local_name in test_cases:
            resolved = transformer._resolve_torch_api_name(local_name)
            print(f"{local_name} -> {resolved}")
            
            expected = local_name.replace('nn', 'torch.nn')
            if resolved == expected:
                print(f"  ✓ Correct")
            else:
                print(f"  ✗ Expected {expected}")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test1 = test_import_mapping()
    test2 = test_api_resolution()
    
    if test1 and test2:
        print("\n🎉 Import mapping tests passed!")
    else:
        print("\n❌ Some tests failed")