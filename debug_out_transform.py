#!/usr/bin/env python3
"""Debug the out parameter transformation"""

try:
    import libcst as cst
    from paconvert.transformer.libcst_transformers.basic_transformer import LibcstBasicTransformer
    from paconvert.global_var import GlobalManager
    
    # Test code
    code = '''import torch
a = [-1, -2, 3]
out = torch.tensor(a)
result = torch.abs(out=out, input=torch.tensor(a))
'''
    
    print("Original code:")
    print(code)
    print("="*50)
    
    # Parse with LibCST
    tree = cst.parse_expression("torch.abs(out=out, input=torch.tensor(a))")
    print(f"Parsed expression: {tree}")
    
    # Check if torch.abs mapping exists
    if "torch.abs" in GlobalManager.API_MAPPING:
        mapping = GlobalManager.API_MAPPING["torch.abs"]
        print(f"torch.abs mapping found:")
        print(f"  paddle_api: {mapping['paddle_api']}")
        print(f"  kwargs_change: {mapping.get('kwargs_change', {})}")
        print(f"  args_list: {mapping.get('args_list', [])}")
    else:
        print("torch.abs mapping not found!")
    
    # Test argument extraction
    if isinstance(tree, cst.Call):
        print(f"\nCall arguments:")
        for i, arg in enumerate(tree.args):
            if isinstance(arg, cst.Arg):
                if arg.keyword:
                    print(f"  {i}: {arg.keyword.value} = {arg.value}")
                else:
                    print(f"  {i}: (positional) {arg.value}")
    
    print("\n✓ Debug completed successfully")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()