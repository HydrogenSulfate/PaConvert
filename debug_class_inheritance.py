#!/usr/bin/env python3
"""
Debug class inheritance issue
"""

import sys
import os

# Add paconvert to path
sys.path.insert(0, os.path.dirname(__file__))

def debug_class_inheritance():
    """Debug the specific class inheritance issue"""
    print("Debugging class inheritance issue...")
    
    # Simple test case
    test_code = '''import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self):
        pass
'''
    
    print("Test code:")
    print(test_code)
    print("="*50)
    
    try:
        from paconvert.backend.libcst_backend import LibcstBackend
        from paconvert.transformer.libcst_transformers.import_transformer import LibcstImportTransformer
        from paconvert.transformer.libcst_transformers.basic_transformer import LibcstBasicTransformer
        from paconvert.api_mapping import API_MAPPING
        
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
        
        # Step 4: Check if nn.Module is in API_MAPPING
        print(f"\nChecking API mappings:")
        print(f"'nn.Module' in API_MAPPING: {'nn.Module' in API_MAPPING}")
        print(f"'torch.nn.Module' in API_MAPPING: {'torch.nn.Module' in API_MAPPING}")
        
        if 'torch.nn.Module' in API_MAPPING:
            print(f"torch.nn.Module mapping: {API_MAPPING['torch.nn.Module']}")
        
        # Step 5: Apply basic transformer with debug
        print("\nApplying basic transformer...")
        basic_transformer = LibcstBasicTransformer(
            tree, "test.py", imports_map, None, {}, {}
        )
        
        # Add debug to the transformer
        original_transform_base_class = basic_transformer._transform_base_class
        def debug_transform_base_class(base_value):
            full_name = basic_transformer.get_full_attr_name(base_value)
            print(f"  Transforming base class: {full_name}")
            
            is_torch = basic_transformer.is_torch_api(base_value)
            print(f"  Is torch API: {is_torch}")
            
            if is_torch:
                torch_api = basic_transformer._resolve_torch_api_name(full_name)
                print(f"  Resolved torch API: {torch_api}")
                
                has_mapping = torch_api in API_MAPPING
                print(f"  Has mapping: {has_mapping}")
                
                if has_mapping:
                    mapping = API_MAPPING[torch_api]
                    print(f"  Mapping: {mapping}")
            
            return original_transform_base_class(base_value)
        
        basic_transformer._transform_base_class = debug_transform_base_class
        
        tree = basic_transformer.transform()
        
        print(f"Torch API count: {basic_transformer.torch_api_count}")
        print(f"Success API count: {basic_transformer.success_api_count}")
        
        # Step 6: Generate final code
        final_code = backend.generate_code(tree)
        print("\n" + "="*50)
        print("Output code:")
        print(final_code)
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_class_inheritance()