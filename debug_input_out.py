#!/usr/bin/env python3
"""Debug input and out parameter handling"""

try:
    import libcst as cst
    
    # Test expressions
    test_expressions = [
        "torch.abs(torch.tensor(a))",
        "torch.abs(input=torch.tensor(a))",
        "torch.abs(torch.tensor(a), out=out)",
        "torch.abs(input=torch.tensor(a), out=out)",
        "torch.abs(out=out, input=torch.tensor(a))"
    ]
    
    for expr in test_expressions:
        print(f"\n=== Testing: {expr} ===")
        
        try:
            tree = cst.parse_expression(expr)
            if isinstance(tree, cst.Call):
                print("Arguments:")
                
                out_found = False
                input_found = False
                positional_count = 0
                
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
                            positional_count += 1
                            print(f"  {i}: (positional) {arg.value}")
                
                print(f"Summary:")
                print(f"  - Positional args: {positional_count}")
                print(f"  - input keyword found: {input_found}")
                print(f"  - out keyword found: {out_found}")
                
                # Expected transformation
                print(f"Expected transformation:")
                if input_found and out_found:
                    print(f"  → paddle.assign(paddle.abs(paddle.to_tensor(a)), output=out)")
                elif input_found:
                    print(f"  → paddle.abs(paddle.to_tensor(a))")
                elif out_found:
                    print(f"  → paddle.assign(paddle.abs(paddle.to_tensor(a)), output=out)")
                else:
                    print(f"  → paddle.abs(paddle.to_tensor(a))")
                    
        except Exception as e:
            print(f"  Error parsing: {e}")
    
    print("\n✅ Debug completed")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()