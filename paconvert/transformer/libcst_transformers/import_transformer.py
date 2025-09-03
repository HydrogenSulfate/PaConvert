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

from typing import Union, Sequence

try:
    import libcst as cst
    from libcst import matchers as m
except ImportError:
    cst = None
    m = None

from .base_transformer import LibcstBaseTransformer
from paconvert.global_var import GlobalManager
from paconvert.utils import log_info


class LibcstImportTransformer(LibcstBaseTransformer):
    """Transformer for handling import statements using libcst."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.paddle_import_added = False
        self.removed_imports = set()
    
    def leave_Import(self, original_node: cst.Import, updated_node: cst.Import) -> Union[cst.Import, cst.RemovalSentinel]:
        """Handle import statements like 'import torch'."""
        new_names = []
        should_remove = False
        
        for name_item in updated_node.names:
            if isinstance(name_item.name, cst.Attribute):
                # Handle dotted imports like torch.nn
                full_name = self._get_dotted_name(name_item.name)
            else:
                full_name = name_item.name.value
            
            if full_name in GlobalManager.TORCH_PACKAGE_MAPPING or full_name.startswith('torch.'):
                # This is a torch import, mark for removal
                should_remove = True
                self._record_torch_import(full_name, name_item.asname)
                if self.logger:
                    log_info(self.logger, f"[{self.file_name}] remove 'import {full_name}'")
            else:
                new_names.append(name_item)
        
        if should_remove:
            self._ensure_paddle_import()
            if not new_names:
                # Remove the entire import statement
                return cst.RemovalSentinel.REMOVE
            else:
                # Keep non-torch imports
                return updated_node.with_changes(names=new_names)
        
        return updated_node
    
    def leave_ImportFrom(self, original_node: cst.ImportFrom, updated_node: cst.ImportFrom) -> Union[cst.ImportFrom, cst.RemovalSentinel]:
        """Handle from imports like 'from torch import nn'."""
        if updated_node.module is None:
            return updated_node
        
        module_name = self._get_dotted_name(updated_node.module)
        
        if module_name in GlobalManager.TORCH_PACKAGE_MAPPING or module_name.startswith('torch.'):
            # This is a torch import
            if isinstance(updated_node.names, cst.ImportStar):
                # Handle 'from torch import *'
                if self.logger:
                    log_info(self.logger, f"[{self.file_name}] remove 'from {module_name} import *'")
                self._record_torch_import(module_name, None)
            else:
                # Handle specific imports
                for name_item in updated_node.names:
                    import_name = name_item.name.value
                    full_import = f"{module_name}.{import_name}"
                    self._record_torch_import(full_import, name_item.asname)
                    if self.logger:
                        log_info(self.logger, f"[{self.file_name}] remove 'from {module_name} import {import_name}'")
            
            self._ensure_paddle_import()
            return cst.RemovalSentinel.REMOVE
        
        return updated_node
    
    def _get_dotted_name(self, node: cst.BaseExpression) -> str:
        """Get the full dotted name from a node."""
        if isinstance(node, cst.Attribute):
            return f"{self._get_dotted_name(node.value)}.{node.attr.value}"
        elif isinstance(node, cst.Name):
            return node.value
        else:
            return str(node)
    
    def _record_torch_import(self, full_name: str, alias: Union[cst.AsName, None]):
        """Record a torch import for later reference."""
        if self.file not in self.imports_map:
            self.imports_map[self.file] = {"torch_packages": []}
        
        # Determine the local name (alias or original)
        if alias and alias.name:
            # Use the alias name (e.g., 'nn' for 'import torch.nn as nn')
            local_name = alias.name.value
        else:
            # For imports without alias
            if '.' not in full_name:
                # Simple import like 'import torch'
                local_name = full_name
            else:
                # Dotted import like 'import torch.nn' -> use 'torch.nn'
                local_name = full_name
        
        # Map local name to full torch name
        self.imports_map[self.file][local_name] = full_name
        
        # Add to torch packages list
        if local_name not in self.imports_map[self.file]["torch_packages"]:
            self.imports_map[self.file]["torch_packages"].append(local_name)
    
    def _ensure_paddle_import(self):
        """Ensure paddle import is added."""
        if not self.paddle_import_added:
            # Create paddle import statement
            paddle_import = cst.SimpleStatementLine(
                body=[cst.Import(names=[cst.ImportAlias(name=cst.Name("paddle"))])]
            )
            self.add_module_insertion(paddle_import)
            self.paddle_import_added = True
            if self.logger:
                log_info(self.logger, f"[{self.file_name}] add 'import paddle' in line 1")