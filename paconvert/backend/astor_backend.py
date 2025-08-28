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
import logging
from typing import List, Any

try:
    import astor
    ASTOR_AVAILABLE = True
except ImportError:
    ASTOR_AVAILABLE = False

from .base import BaseBackend


class AstorBackend(BaseBackend):
    """AST backend using the standard ast module and astor for code generation."""
    
    def __init__(self):
        """Initialize AstorBackend."""
        super().__init__()
        if not ASTOR_AVAILABLE:
            raise ImportError("astor is required for AstorBackend but not available")
    
    def parse_code(self, code: str) -> ast.AST:
        """Parse source code using ast.parse.
        
        Args:
            code: Source code string to parse
            
        Returns:
            AST representation
            
        Raises:
            SyntaxError: If code cannot be parsed
        """
        try:
            return ast.parse(code)
        except SyntaxError as e:
            raise SyntaxError(f"Failed to parse code: {e}")
    
    def generate_code(self, tree: ast.AST) -> str:
        """Generate source code using astor.to_source.
        
        Args:
            tree: AST representation to convert back to code
            
        Returns:
            Generated source code string
            
        Raises:
            ValueError: If tree cannot be converted to code
        """
        try:
            return astor.to_source(tree)
        except Exception as e:
            raise ValueError(f"Failed to generate code from AST: {e}")
    
    def create_transformers(self, tree: ast.AST, file: str, imports_map: dict,
                          logger: logging.Logger, all_api_map: dict = None,
                          unsupport_api_map: dict = None) -> List[Any]:
        """Create AST transformers for the astor backend.
        
        Args:
            tree: AST tree to transform
            file: File path being processed
            imports_map: Import mapping dictionary
            logger: Logger instance
            all_api_map: Optional API mapping dictionary
            unsupport_api_map: Optional unsupported API mapping dictionary
            
        Returns:
            List of transformer instances
        """
        # Import transformer classes here to avoid circular imports
        from paconvert.transformer.import_transformer import ImportTransformer
        from paconvert.transformer.tensor_requires_grad_transformer import (
            TensorRequiresGradTransformer,
        )
        from paconvert.transformer.basic_transformer import BasicTransformer
        from paconvert.transformer.custom_op_transformer import (
            PreCustomOpTransformer,
            CustomOpTransformer,
        )
        
        # Create transformer instances - same order as in original converter
        transformers = [
            ImportTransformer,  # import ast transformer
            TensorRequiresGradTransformer,  # attribute requires_grad transformer
            BasicTransformer,  # most of api transformer
            PreCustomOpTransformer,  # pre process for C++ custom op
            CustomOpTransformer,  # C++ custom op transformer
        ]
        
        transformer_instances = []
        for transformer_class in transformers:
            transformer_instance = transformer_class(
                tree,
                file,
                imports_map,
                logger,
                all_api_map,
                unsupport_api_map,
            )
            transformer_instances.append(transformer_instance)
        
        return transformer_instances
    
    def is_available(self) -> bool:
        """Check if astor backend is available.
        
        Returns:
            True if astor is available, False otherwise
        """
        return ASTOR_AVAILABLE
    
    def get_backend_name(self) -> str:
        """Get the name of this backend.
        
        Returns:
            Backend name string
        """
        return "astor"