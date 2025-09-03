#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

def ultimate_debug():
    """Ultimate debug to find the issue"""
    print("Ultimate debug...")
    
    from paconvert.backend.libcst_backend import LibcstBackend
    from paconvert.transformer.libcst_transformers.import_transformer import LibcstImportTransformer
    from paconvert.transformer.libcst_transformers.basic_transformer import LibcstBasicTransformer
    from paconvert.api_mapping import API_MAPPING
    import libcst as cst
    
    # Simple test
    code = '''import torch.nn as nn
class Test(nn.Module):
    pass'''
    
    print("Code:")
    print(code)
    print("="*50)
    
    # Step 1: Parse
    backend = LibcstBackend()
    tree = backend.parse_code(code)
    print("✓ Parsed")
    
    # Step 2: Import transformer
    imports_map = {}
    import_transformer = LibcstImportTransformer(tree, "test.py", imports_map, None, {}, {})
    tree = import_transformer.transform()
    print(f"✓ Import transformer applied, imports_map: {imports_map}")
    
    # Step 3: Check if nn.Module is correctly mapped
    if 'test.py' in imports_map and 'nn' in imports_map['test.py']:
        print(f"✓ nn mapped to: {imports_map['test.py']['nn']}")
    else:
        print("✗ nn not properly mapped")
        return False
    
    # Step 4: Check API mapping
    if 'torch.nn.Module' in API_MAPPING:
        print(f"✓ torch.nn.Module in API_MAPPING: {API_MAPPING['torch.nn.Module']}")
    else:
        print("✗ torch.nn.Module not in API_MAPPING")
        return False
    
    # Step 5: Create basic transformer with custom debug
    class DebugBasicTransformer(LibcstBasicTransformer):
        def leave_ClassDef(self, original_node, updated_node):
            print(f"\n--- Processing class: {updated_node.name.value} ---")
            if updated_node.bases:
                for i, base in enumerate(updated_node.bases):
                    print(f"Base {i}: {type(base)} - {base}")
                    if isinstance(base, cst.Arg) and base.value:
                        print(f"  Processing base.value: {type(base.value)}")
                        result = self._transform_base_class(base.value)
                        print(f"  Transform result: {type(result)} - {result}")
                        if result != base.value:
                            print("  ✓ Base class was transformed!")
                            return updated_node.with_changes(bases=[base.with_changes(value=result)])
                        else:
                            print("  ✗ Base class was not transformed")
            return updated_node
        
        def _transform_base_class(self, base_value):
            print(f"    _transform_base_class called with: {type(base_value)} - {base_value}")
            full_name = self.get_full_attr_name(base_value)
            print(f"    Full name: {full_name}")
            
            is_torch = self.is_torch_api(base_value)
            print(f"    Is torch API: {is_torch}")
            
            if not is_torch:
                print("    Not torch API, returning original")
                return base_value
            
            resolved = self._resolve_torch_api_name(full_name)
            print(f"    Resolved name: {resolved}")
            
            has_mapping = resolved in API_MAPPING
            print(f"    Has mapping: {has_mapping}")
            
            if not has_mapping:
                print("    No mapping, returning original")
                return base_value
            
            mapping = API_MAPPING[resolved]
            paddle_api = mapping.get("paddle_api")
            print(f"    Paddle API: {paddle_api}")
            
            if not paddle_api:
                print("    No paddle API, returning original")
                return base_value
            
            new_base = self._create_paddle_func_ref(paddle_api)
            print(f"    Created new base: {type(new_base)} - {new_base}")
            
            self.torch_api_count += 1
            self.success_api_count += 1
            
            return new_base
    
    # Apply debug transformer
    debug_transformer = DebugBasicTransformer(tree, "test.py", imports_map, None, {}, {})
    tree = debug_transformer.transform()
    
    print(f"\nTransformer stats:")
    print(f"  Torch API count: {debug_transformer.torch_api_count}")
    print(f"  Success API count: {debug_transformer.success_api_count}")
    
    # Generate result
    result = backend.generate_code(tree)
    print("\nFinal result:")
    print(result)
    
    # Check result
    if "paddle.nn.Layer" in result:
        print("✓ SUCCESS: Base class was converted!")
        return True
    else:
        print("✗ FAILED: Base class was not converted")
        return False

if __name__ == "__main__":
    ultimate_debug()