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
    import astor
except ImportError:
    astor = None

from .base_backend import BaseBackend


class AstorBackend(BaseBackend):
    """Backend implementation using astor for AST to source conversion."""
    
    def __init__(self):
        if astor is None:
            raise ImportError("astor is required for AstorBackend. Install with: pip install astor")
    
    def parse_code(self, code: str) -> ast.AST:
        """Parse source code into an AST using ast.parse."""
        return ast.parse(code)
    
    def generate_code(self, tree: ast.AST) -> str:
        """Generate source code from AST using astor.to_source."""
        return astor.to_source(tree)
    
    def create_transformers(self) -> List[Type]:
        """Return existing transformer classes that work with ast module."""
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
        """Convert a single AST node to source code using astor."""
        return astor.to_source(node).replace("\n", "")
    
    def get_backend_type(self) -> str:
        """Return the backend type identifier."""
        return "ast"