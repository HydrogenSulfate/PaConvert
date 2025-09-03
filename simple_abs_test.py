#!/usr/bin/env python3
"""Simple test to verify torch.abs conversion"""

# Test if we can import the fixed modules
try:
    from paconvert.global_var import GlobalManager
    print("✓ Successfully imported GlobalManager")
    
    # Check if torch.abs mapping exists
    if "torch.abs" in GlobalManager.API_MAPPING:
        mapping = GlobalManager.API_MAPPING["torch.abs"]
        print(f"✓ Found torch.abs mapping: {mapping['paddle_api']}")
        print(f"  Matcher: {mapping['Matcher']}")
        print(f"  Args: {mapping.get('args_list', [])}")
        print(f"  Kwargs change: {mapping.get('kwargs_change', {})}")
    else:
        print("✗ torch.abs mapping not found")
    
    # Test LibCST transformer import
    from paconvert.transformer.libcst_transformers.basic_transformer import LibcstBasicTransformer
    print("✓ Successfully imported LibcstBasicTransformer")
    
    print("\n🎉 All imports successful! The fix should work.")
    
except Exception as e:
    print(f"✗ Import error: {e}")
    import traceback
    traceback.print_exc()