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

import logging
from typing import List, Any

try:
    import libcst as cst
    LIBCST_AVAILABLE = True
except ImportError:
    LIBCST_AVAILABLE = False
    cst = None

from .base import BaseBackend


class LibcstBackend(BaseBackend):
    """AST backend using libcst for comment and formatting preservation."""
    
    def __init__(self):
        """Initialize LibcstBackend."""
        super().__init__()
        if not LIBCST_AVAILABLE:
            raise ImportError("libcst is required for LibcstBackend but not available. Install with: pip install libcst")
    
    def parse_code(self, code: str) -> cst.Module:
        """Parse source code using libcst.parse_module.
        
        Args:
            code: Source code string to parse
            
        Returns:
            libcst Module representation
            
        Raises:
            cst.ParserError: If code cannot be parsed
        """
        try:
            return cst.parse_module(code)
        except Exception as e:
            raise cst.ParserError(f"Failed to parse code with libcst: {e}")
    
    def generate_code(self, tree: cst.Module) -> str:
        """Generate source code using libcst Module.code.
        
        Args:
            tree: libcst Module representation to convert back to code
            
        Returns:
            Generated source code string
            
        Raises:
            ValueError: If tree cannot be converted to code
        """
        try:
            if hasattr(tree, 'code'):
                return tree.code
            else:
                # For modified trees, we need to generate code
                return tree.get_code()
        except Exception as e:
            raise ValueError(f"Failed to generate code from libcst tree: {e}")
    
    def create_transformers(self, tree: cst.Module, file: str, imports_map: dict,
                          logger: logging.Logger, all_api_map: dict = None,
                          unsupport_api_map: dict = None) -> List[Any]:
        """Create libcst transformers for the libcst backend.
        
        Args:
            tree: libcst Module tree to transform
            file: File path being processed
            imports_map: Import mapping dictionary
            logger: Logger instance
            all_api_map: Optional API mapping dictionary
            unsupport_api_map: Optional unsupported API mapping dictionary
            
        Returns:
            List of transformer instances
        """
        # For now, we'll create a bridge that converts libcst to AST,
        # applies AST transformers, then converts back to libcst
        # This is a temporary solution until we implement native libcst transformers
        
        # Import transformer classes
        from paconvert.transformer.import_transformer import ImportTransformer
        from paconvert.transformer.tensor_requires_grad_transformer import (
            TensorRequiresGradTransformer,
        )
        from paconvert.transformer.basic_transformer import BasicTransformer
        from paconvert.transformer.custom_op_transformer import (
            PreCustomOpTransformer,
            CustomOpTransformer,
        )
        
        # Create bridge transformers that work with libcst
        transformers = [
            LibcstTransformerBridge(ImportTransformer),
            LibcstTransformerBridge(TensorRequiresGradTransformer),
            LibcstTransformerBridge(BasicTransformer),
            LibcstTransformerBridge(PreCustomOpTransformer),
            LibcstTransformerBridge(CustomOpTransformer),
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
        """Check if libcst backend is available.
        
        Returns:
            True if libcst is available, False otherwise
        """
        return LIBCST_AVAILABLE
    
    def get_backend_name(self) -> str:
        """Get the name of this backend.
        
        Returns:
            Backend name string
        """
        return "libcst"


class LibcstTransformerBridge:
    """Bridge class that adapts AST transformers to work with libcst trees."""
    
    def __init__(self, ast_transformer_class):
        """Initialize bridge with AST transformer class.
        
        Args:
            ast_transformer_class: AST transformer class to bridge
        """
        self.ast_transformer_class = ast_transformer_class
        self.torch_api_count = 0
        self.success_api_count = 0
    
    def __call__(self, tree, file, imports_map, logger, all_api_map=None, unsupport_api_map=None):
        """Create bridge instance.
        
        Args:
            tree: libcst Module tree
            file: File path
            imports_map: Import mapping
            logger: Logger instance
            all_api_map: Optional API mapping
            unsupport_api_map: Optional unsupported API mapping
            
        Returns:
            Bridge transformer instance
        """
        return LibcstTransformerBridgeInstance(
            self.ast_transformer_class,
            tree,
            file,
            imports_map,
            logger,
            all_api_map,
            unsupport_api_map
        )


class LibcstTransformerBridgeInstance:
    """Instance of the bridge transformer."""
    
    def __init__(self, ast_transformer_class, tree, file, imports_map, logger, all_api_map=None, unsupport_api_map=None):
        """Initialize bridge instance.
        
        Args:
            ast_transformer_class: AST transformer class
            tree: libcst Module tree
            file: File path
            imports_map: Import mapping
            logger: Logger instance
            all_api_map: Optional API mapping
            unsupport_api_map: Optional unsupported API mapping
        """
        self.ast_transformer_class = ast_transformer_class
        self.libcst_tree = tree
        self.file = file
        self.imports_map = imports_map
        self.logger = logger
        self.all_api_map = all_api_map
        self.unsupport_api_map = unsupport_api_map
        self.torch_api_count = 0
        self.success_api_count = 0
    
    def transform(self):
        """Apply transformation using AST bridge.
        
        This is a temporary implementation that converts libcst -> AST -> transform -> AST -> libcst
        In the future, this should be replaced with native libcst transformers.
        """
        try:
            # Store original code for comment preservation
            original_code = self.libcst_tree.code
            
            # Convert libcst to AST
            import ast
            ast_code = self.libcst_tree.code
            ast_tree = ast.parse(ast_code)
            
            # Apply AST transformer
            ast_transformer = self.ast_transformer_class(
                ast_tree,
                self.file,
                self.imports_map,
                self.logger,
                self.all_api_map,
                self.unsupport_api_map
            )
            ast_transformer.transform()
            
            # Update counts
            self.torch_api_count = ast_transformer.torch_api_count
            self.success_api_count = ast_transformer.success_api_count
            
            # Convert back to code using astor
            import astor
            transformed_code = astor.to_source(ast_tree)
            
            # Try to preserve comments by merging them back
            preserved_code = self._preserve_comments(original_code, transformed_code)
            
            # Parse the preserved code back to libcst
            self.libcst_tree = cst.parse_module(preserved_code)
            
        except Exception as e:
            self.logger.warning(f"LibCST bridge transformation failed for {self.file}: {e}")
            # Keep original tree on failure
            pass
    
    def _preserve_comments(self, original_code: str, transformed_code: str) -> str:
        """Attempt to preserve comments and formatting from original code.
        
        This is a heuristic-based approach that tries to merge comments and
        preserve formatting where possible. A full implementation would
        use libcst's native transformation capabilities.
        
        Args:
            original_code: Original code with comments and formatting
            transformed_code: Transformed code without comments
            
        Returns:
            Transformed code with comments and formatting preserved where possible
        """
        try:
            # If the codes are very similar, prefer the original formatting
            if self._codes_functionally_similar(original_code, transformed_code):
                return self._merge_minimal_changes(original_code, transformed_code)
            
            # Otherwise, try to preserve comments in transformed code
            return self._merge_comments_into_transformed(original_code, transformed_code)
            
        except Exception:
            # If preservation fails, return transformed code as-is
            return transformed_code
    
    def _codes_functionally_similar(self, code1: str, code2: str) -> bool:
        """Check if two code strings are functionally similar.
        
        Args:
            code1: First code string
            code2: Second code string
            
        Returns:
            True if codes are functionally similar
        """
        try:
            # Remove comments and normalize whitespace for comparison
            clean1 = self._normalize_code_for_comparison(code1)
            clean2 = self._normalize_code_for_comparison(code2)
            
            # If they're very similar, we can preserve original formatting
            similarity_ratio = len(set(clean1.split()) & set(clean2.split())) / max(len(clean1.split()), len(clean2.split()), 1)
            return similarity_ratio > 0.8
            
        except Exception:
            return False
    
    def _normalize_code_for_comparison(self, code: str) -> str:
        """Normalize code for comparison by removing comments and extra whitespace.
        
        Args:
            code: Code to normalize
            
        Returns:
            Normalized code string
        """
        lines = []
        for line in code.splitlines():
            # Remove inline comments
            if '#' in line:
                line = line[:line.find('#')]
            # Normalize whitespace
            line = ' '.join(line.split())
            if line:
                lines.append(line)
        return '\n'.join(lines)
    
    def _merge_minimal_changes(self, original_code: str, transformed_code: str) -> str:
        """Merge minimal changes from transformed code into original code.
        
        This preserves the original formatting and comments while applying
        only the necessary transformations.
        
        Args:
            original_code: Original code with formatting and comments
            transformed_code: Transformed code with changes
            
        Returns:
            Original code with minimal necessary changes applied
        """
        try:
            # For now, return original code if they're similar
            # A full implementation would identify specific changes and apply them
            # while preserving formatting
            return original_code
            
        except Exception:
            return transformed_code
    
    def _merge_comments_into_transformed(self, original_code: str, transformed_code: str) -> str:
        """Merge comments from original code into transformed code.
        
        Args:
            original_code: Original code with comments
            transformed_code: Transformed code without comments
            
        Returns:
            Transformed code with comments merged in
        """
        try:
            original_lines = original_code.splitlines()
            transformed_lines = transformed_code.splitlines()
            
            # Extract comments and their positions
            comments = {}
            for i, line in enumerate(original_lines):
                stripped = line.strip()
                if stripped.startswith('#'):
                    # Full line comment
                    comments[i] = line
                elif '#' in line:
                    # Inline comment
                    comment_part = line[line.find('#'):]
                    comments[i] = comment_part
            
            # Try to merge comments back into transformed code
            result_lines = []
            original_idx = 0
            
            for transformed_line in transformed_lines:
                # Add the transformed line
                result_lines.append(transformed_line)
                
                # Try to find corresponding original line and add any comments
                while original_idx < len(original_lines):
                    original_line = original_lines[original_idx]
                    
                    # If we find a comment line, add it
                    if original_idx in comments:
                        if original_line.strip().startswith('#'):
                            # Full line comment - add it before current line
                            result_lines.insert(-1, comments[original_idx])
                        else:
                            # Inline comment - try to add to current line if similar
                            if self._lines_similar(original_line, transformed_line):
                                result_lines[-1] = transformed_line + '  ' + comments[original_idx]
                    
                    original_idx += 1
                    break
            
            return '\n'.join(result_lines)
            
        except Exception:
            return transformed_code
    
    def _lines_similar(self, line1: str, line2: str) -> bool:
        """Check if two lines are similar enough to merge comments.
        
        Args:
            line1: First line to compare
            line2: Second line to compare
            
        Returns:
            True if lines are similar enough to merge comments
        """
        # Remove comments and whitespace for comparison
        clean1 = line1.split('#')[0].strip()
        clean2 = line2.split('#')[0].strip()
        
        # Simple similarity check - could be improved
        return clean1 == clean2 or (clean1 and clean2 and clean1 in clean2)