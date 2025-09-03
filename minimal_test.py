#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

# Minimal test
from paconvert.backend.libcst_backend import LibcstBackend
from paconvert.transformer.libcst_transformers.import_transformer import LibcstImportTransformer
from paconvert.transformer.libcst_transformers.basic_transformer import LibcstBasicTransformer
import libcst as cst

# Test code
code = '''import torch.nn as nn
class Test(nn.Module):
    pass'''

print("Code:", code)

# Parse
backend = LibcstBackend()
tree = backend.parse_code(code)

# Apply import transformer
imports_map = {}
import_transformer = LibcstImportTransformer(tree, "test.py", imports_map, None, {}, {})
tree = import_transformer.transform()

print("Imports map:", imports_map)

# Apply basic transformer
basic_transformer = LibcstBasicTransformer(tree, "test.py", imports_map, None, {}, {})

# Check if the class definition is being processed
class DebugTransformer(LibcstBasicTransformer):
    def leave_ClassDef(self, original_node, updated_node):
        print("Processing class definition:", updated_node.name.value)
        if updated_node.bases:
            for i, base in enumerate(updated_node.bases):
                print(f"  Base {i}: {base}")
                if isinstance(base, cst.Arg) and base.value:
                    full_name = self.get_full_attr_name(base.value)
                    print(f"    Full name: {full_name}")
                    is_torch = self.is_torch_api(base.value)
                    print(f"    Is torch: {is_torch}")
                    if is_torch:
                        resolved = self._resolve_torch_api_name(full_name)
                        print(f"    Resolved: {resolved}")
        return super().leave_ClassDef(original_node, updated_node)

debug_transformer = DebugTransformer(tree, "test.py", imports_map, None, {}, {})
tree = debug_transformer.transform()

# Generate result
result = backend.generate_code(tree)
print("Result:")
print(result)