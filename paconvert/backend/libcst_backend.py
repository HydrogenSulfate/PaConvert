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

from typing import Any, List, Type, Union, Optional

try:
    import libcst as cst
    from libcst.metadata import MetadataWrapper
    from libcst import matchers as m
except ImportError:
    cst = None
    MetadataWrapper = None
    m = None

from .base_backend import BaseBackend


class LibcstBackend(BaseBackend):
    """Backend implementation using native libcst for CST manipulation."""
    
    def __init__(self):
        if cst is None:
            raise ImportError("libcst is required for LibcstBackend. Install with: pip install libcst")
    
    def parse_code(self, code: str) -> cst.Module:
        """Parse source code into a CST using libcst.parse_module."""
        return cst.parse_module(code)
    
    def generate_code(self, tree: cst.Module) -> str:
        """Generate source code from CST using libcst."""
        return tree.code
    
    def create_transformers(self) -> List[Type]:
        """Return transformer classes adapted for libcst backend."""
        from paconvert.transformer.libcst_transformers.import_transformer import LibcstImportTransformer
        from paconvert.transformer.libcst_transformers.basic_transformer import LibcstBasicTransformer
        from paconvert.transformer.libcst_transformers.tensor_requires_grad_transformer import LibcstTensorRequiresGradTransformer
        from paconvert.transformer.libcst_transformers.custom_op_transformer import (
            LibcstPreCustomOpTransformer,
            LibcstCustomOpTransformer,
        )
        
        return [
            LibcstImportTransformer,
            LibcstTensorRequiresGradTransformer,
            LibcstBasicTransformer,
            LibcstPreCustomOpTransformer,
            LibcstCustomOpTransformer,
        ]
    
    def node_to_source(self, node: cst.CSTNode) -> str:
        """Convert a single CST node to source code."""
        if isinstance(node, cst.Module):
            return node.code
        else:
            # For non-module nodes, we need to create a temporary module
            # to get the source code
            try:
                if isinstance(node, cst.BaseExpression):
                    temp_module = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=node)])])
                elif isinstance(node, cst.BaseStatement):
                    if isinstance(node, cst.SimpleStatementLine):
                        temp_module = cst.Module(body=[node])
                    else:
                        temp_module = cst.Module(body=[node])
                else:
                    # For other node types, try to wrap in an expression
                    temp_module = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=node)])])
                
                code = temp_module.code.strip()
                # Remove trailing newline if present
                if code.endswith('\n'):
                    code = code[:-1]
                return code
            except Exception:
                # Fallback: return string representation
                return str(node)
    
    def get_backend_type(self) -> str:
        """Return the backend type identifier."""
        return "cst"