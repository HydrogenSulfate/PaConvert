#!/usr/bin/env python3
"""Debug argument transformation logic"""

try:
    import libcst as cst
    from paconvert.global_var import GlobalManager
    
    # Check torch.abs mapping
    if "torch.abs" in GlobalManager.API_MAPPING:
        mapping = GlobalManager.API_MAPPING["torch.abs"]
        print("torch.abs mapping:")
        print(f"  paddle_api: {mapping['paddle_api']}")
        print(f"  kwargs_change: {mapping.get('kwargs_change', {})}")
        print(f"  args_list: {mapping.get('args_list', [])}")
        
        # Test argument parsing
        test_expressions = [
            "torch.abs(torch.tensor(a), out=out)",
            "torch.abs(input=torch.tensor(a), out=out)",
            "torch.abs(out=out, input=torch.tensor(a))"
        ]
        
        for expr in test_expressions:
            print(f"\nTesting: {expr}")
            try:
                tree = cst.parse_expression(expr)
                if isinstance(tree, cst.Call):
                    print("Arguments:")
                    out_found = False
                    input_found = False
                    for i, arg in enumerate(tree.args):
                        if isinstance(arg, cst.Arg):
                            if arg.keyword:
                                keyword = arg.keyword.value
                                print(f"  {i}: {keyword} = {arg.value}")
                                if keyword == "out":
                                    out_found = True
                                elif keyword == "input":
                                    input_found = True
                            else:
                                print(f"  {i}: (positional) {arg.value}")
                    
                    print(f"  out parameter found: {out_found}")
                    print(f"  input parameter found: {input_found}")
                    
                    # Test kwargs_change logic
                    kwargs_change = mapping.get("kwargs_change", {})
                    print(f"  kwargs_change: {kwargs_change}")
                    
                    if input_found:
                        if "input" in kwargs_change:
                            new_name = kwargs_change["input"]
                            print(f"  input should be renamed to: '{new_name}'")
                        else:
                            print("  input parameter not in kwargs_change")
                    
                    if out_found:
                        if "out" in kwargs_change:
                            new_name = kwargs_change["out"]
                            print(f"  out should be renamed to: '{new_name}'")
                        else:
                            print("  out parameter not in kwargs_change - should be handled specially")
                            
            except Exception as e:
                print(f"  Error parsing: {e}")
    else:
        print("torch.abs mapping not found!")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()