#!/usr/bin/env python3
"""
Debug script to test libcst backend step by step
"""

import sys
import os

# Add paconvert to path
sys.path.insert(0, os.path.dirname(__file__))

def debug_libcst_conversion():
    """Debug the libcst conversion process step by step"""
    print("Debugging LibCST conversion...")
    
    # Test code
    test_code = '''import torch
x = torch.tensor([1, 2, 3])
y = torch.zeros(3, 4)'''
    
    print("Original code:")
    print(test_code)
    print("\n" + "="*50)
    
    try:
        from paconvert.backend.libcst_backend import LibcstBackend
        import libcst as cst
        
        # Step 1: Create backend
        backend = LibcstBackend()
        print("✓ LibCST backend created")
        
        # Step 2: Parse code
        tree = backend.parse_code(test_code)
        print("✓ Code parsed successfully")
        print(f"Tree type: {type(tree)}")
        
        # Step 3: Test transformers
        transformers = backend.create_transformers()
        print(f"✓ Created {len(transformers)} transformers:")
        for t in transformers:
            print(f"  - {t.__name__}")
        
        # Step 4: Create imports map
        imports_map = {}
        
        # Step 5: Apply transformers one by one
        current_tree = tree
        for i, transformer_class in enumerate(transformers):
            print(f"\nApplying transformer {i+1}: {transformer_class.__name__}")
            
            transformer = transformer_class(
                current_tree,
                "test_file.py",
                imports_map,
                None,  # logger
                {},    # all_api_map
                {}     # unsupport_api_map
            )
            
            # Transform
            new_tree = transformer.transform()
            print(f"  Torch API count: {transformer.torch_api_count}")
            print(f"  Success API count: {transformer.success_api_count}")
            
            # Check if tree changed
            if new_tree != current_tree:
                print("  ✓ Tree was modified")
                current_tree = new_tree
            else:
                print("  - Tree unchanged")
        
        # Step 6: Generate final code
        final_code = backend.generate_code(current_tree)
        print("\n" + "="*50)
        print("Final code:")
        print(final_code)
        
        # Step 7: Check imports map
        print("\n" + "="*50)
        print("Imports map:")
        print(imports_map)
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_libcst_conversion()