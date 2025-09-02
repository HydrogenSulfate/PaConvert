# Copyright (c) 2022  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ast
from typing import Any, List, Type

try:
    import libcst as cst
    from libcst.metadata import MetadataWrapper
except ImportError:
    cst = None
    MetadataWrapper = None

from .base_backend import BaseBackend


class LibcstBackend(BaseBackend):
    """Backend implementation using libcst for CST to source conversion."""
    
    def __init__(self):
        if cst is None:
            raise ImportError("libcst is required for LibcstBackend. Install with: pip install libcst")
    
    def parse_code(self, code: str) -> Any:
        """Parse source code into a CST using libcst.parse_module."""
        # For compatibility with existing transformers, we still need to return an AST
        # The CST will be used internally for code generation
        self._cst_tree = cst.parse_module(code)
        self._original_code = code
        return ast.parse(code)
    
    def generate_code(self, tree: ast.AST) -> str:
        """Generate source code from AST by converting back to CST and using libcst."""
        # For now, we'll use a hybrid approach:
        # 1. Convert the modified AST back to source using ast.unparse (Python 3.9+) or a fallback
        # 2. Parse with libcst to preserve formatting where possible
        try:
            # Try using ast.unparse if available (Python 3.9+)
            if hasattr(ast, 'unparse'):
                intermediate_code = ast.unparse(tree)
            else:
                # Fallback to astor for older Python versions
                import astor
                intermediate_code = astor.to_source(tree)
            
            # Parse with libcst to get better formatting
            cst_tree = cst.parse_module(intermediate_code)
            return cst_tree.code
        except Exception:
            # If libcst parsing fails, fall back to basic conversion
            if hasattr(ast, 'unparse'):
                return ast.unparse(tree)
            else:
                import astor
                return astor.to_source(tree)
    
    def create_transformers(self) -> List[Type]:
        """Return transformer classes adapted for libcst backend."""
        # For now, we'll use the same transformers as astor backend
        # In the future, these could be replaced with native libcst transformers
        from paconvert.transformer.basic_transformer import BasicTransformer
        from paconvert.transformer.import_transformer import ImportTransformer
        from paconvert.transformer.tensor_requires_grad_transformer import (
            TensorRequiresGradTransformer,
        )
        from paconvert.transformer.custom_op_transformer import (
            PreCustomOpTransformer,
            CustomOpTransformer,
        )
        
        return [
            ImportTransformer,
            TensorRequiresGradTransformer,
            BasicTransformer,
            PreCustomOpTransformer,
            CustomOpTransformer,
        ]
    
    def node_to_source(self, node: ast.AST) -> str:
        """Convert a single AST node to source code."""
        try:
            if hasattr(ast, 'unparse'):
                return ast.unparse(node).replace("\n", "")
            else:
                import astor
                return astor.to_source(node).replace("\n", "")
        except Exception:
            # Fallback for complex nodes
            return str(node)