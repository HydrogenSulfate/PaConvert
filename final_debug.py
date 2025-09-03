#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

def final_debug():
    """Final debug to find the exact issue"""
    print("Final debug for class inheritance...")
    
    test_code = '''import torch.nn as nn

class ConvNet(nn.Module):
    pass
'''
    
    try:
        from paconvert.backend.libcst_backend import LibcstBackend
        from paconvert.transformer.libcst_transformers.import_transformer import LibcstImportTransformer
        from paconvert.transformer.libcst_transformers.basic_transformer import LibcstBasicTransformer
        from paconvert.api_mapping import API_MAPPING
        import libcst as cst
        
        print("Step 1: Parse code")
        backend = LibcstBackend()
        tree = backend.parse_code(test_code)
        print("✓ Parsed")
        
        print("\nStep 2: Apply import transformer")
        imports_map = {}
        import_transformer = LibcstImportTransformer(
            tree, "test.py", imports_map, None, {}, {}
        )
        tree = import_transformer.transform()
        print(f"Imports map: {imports_map}")
        
        print("\nStep 3: Check API mapping")
        print(f"'torch.nn.Module' in API_MAPPING: {'torch.nn.Module' in API_MAPPING}")
        if 'torch.nn.Module' in API_MAPPING:
            print(f"Mapping: {API_MAPPING['torch.nn.Module']}")
        
        print("\nStep 4: Manual test of resolution")
        basic_transformer = LibcstBasicTransformer(
            tree, "test.py", imports_map, None, {}, {}
        )
        
        # Create a test node for nn.Module
        nn_module_node = cst.Attribute(
            value=cst.Name("nn"),
            attr=cst.Name("Module")
        )
        
        full_name = basic_transformer.get_full_attr_name(nn_module_node)
        print(f"Full name: {full_name}")
        
        is_torch = basic_transformer.is_torch_api(nn_module_node)
        print(f"Is torch API: {is_torch}")
        
        if is_torch:
            resolved = basic_transformer._resolve_torch_api_name(full_name)
            print(f"Resolved name: {resolved}")
            
            has_mapping = resolved in API_MAPPING
            print(f"Has mapping: {has_mapping}")
        
        print("\nStep 5: Apply basic transformer")
        tree = basic_transformer.transform()
        
        print(f"Torch API count: {basic_transformer.torch_api_count}")
        print(f"Success API count: {basic_transformer.success_api_count}")
        
        print("\nStep 6: Generate code")
        final_code = backend.generate_code(tree)
        print("Final code:")
        print(final_code)
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    final_debug()