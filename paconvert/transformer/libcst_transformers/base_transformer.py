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

import collections
import os
import re
from typing import Dict, Any, Optional, List, Union

try:
    import libcst as cst
    from libcst import matchers as m
    from libcst.metadata import MetadataWrapper, ScopeProvider
except ImportError:
    cst = None
    m = None
    MetadataWrapper = None
    ScopeProvider = None

from paconvert.utils import UtilsFileHelper, log_debug


class LibcstBaseTransformer(cst.CSTTransformer):
    """Base transformer for libcst-based transformations."""
    
    def __init__(
        self, 
        root: cst.Module, 
        file: str, 
        imports_map: Dict, 
        logger, 
        all_api_map: Optional[Dict] = None, 
        unsupport_api_map: Optional[Dict] = None
    ):
        super().__init__()
        self.root = root
        self.file = file
        self.file_name = os.path.basename(file)
        self.imports_map = imports_map
        self.torch_api_count = 0
        self.success_api_count = 0
        self.logger = logger
        self.all_api_map = all_api_map or collections.defaultdict(dict)
        self.unsupport_api_map = unsupport_api_map or collections.defaultdict(int)
        
        # Track nodes to insert at module level
        self.module_insertions: List[cst.CSTNode] = []
        
        # Track torch packages for this file
        if file not in self.imports_map:
            self.imports_map[file] = {"torch_packages": []}
    
    def transform(self) -> cst.Module:
        """Transform the CST and return the modified tree."""
        # Use metadata wrapper for scope analysis if needed
        wrapper = MetadataWrapper(self.root)
        transformed = self.root.visit(self)
        
        # Insert any module-level nodes
        if self.module_insertions:
            new_body = list(transformed.body)
            # Insert at the beginning after any existing imports
            insert_index = 0
            for i, stmt in enumerate(new_body):
                if not self._is_import_statement(stmt):
                    insert_index = i
                    break
            else:
                insert_index = len(new_body)
            
            for insertion in reversed(self.module_insertions):
                new_body.insert(insert_index, insertion)
            
            transformed = transformed.with_changes(body=new_body)
        
        return transformed
    
    def _is_import_statement(self, stmt: cst.CSTNode) -> bool:
        """Check if a statement is an import statement."""
        if isinstance(stmt, cst.SimpleStatementLine):
            for substmt in stmt.body:
                if isinstance(substmt, (cst.Import, cst.ImportFrom)):
                    return True
        return False
    
    def node_to_source(self, node: cst.CSTNode) -> str:
        """Convert a CST node to source code."""
        if isinstance(node, cst.Module):
            return node.code
        else:
            try:
                if isinstance(node, cst.BaseExpression):
                    temp_module = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=node)])])
                elif isinstance(node, cst.BaseStatement):
                    temp_module = cst.Module(body=[node])
                else:
                    temp_module = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=node)])])
                
                code = temp_module.code.strip()
                if code.endswith('\n'):
                    code = code[:-1]
                return code
            except Exception:
                return str(node)
    
    def get_full_attr_name(self, node: cst.BaseExpression) -> str:
        """Get the full attribute name from a node (e.g., torch.nn.Linear)."""
        if isinstance(node, cst.Attribute):
            return f"{self.get_full_attr_name(node.value)}.{node.attr.value}"
        elif isinstance(node, cst.Name):
            return node.value
        else:
            return "Unknown"
    
    def is_torch_api(self, node: cst.BaseExpression) -> bool:
        """Check if a node represents a torch API call."""
        full_name = self.get_full_attr_name(node)
        
        # Check if this file has torch packages recorded
        if self.file not in self.imports_map or "torch_packages" not in self.imports_map[self.file]:
            return False
        
        torch_packages = self.imports_map[self.file]["torch_packages"]
        
        # Debug print
        if self.logger:
            log_debug(self.logger, f"Checking if {full_name} is torch API, packages: {torch_packages}")
        
        for package in torch_packages:
            if full_name.startswith(f"{package}.") or full_name == package:
                if self.logger:
                    log_debug(self.logger, f"  ✓ {full_name} matches package {package}")
                return True
        
        if self.logger:
            log_debug(self.logger, f"  ✗ {full_name} does not match any torch package")
        return False
    
    def add_module_insertion(self, node: cst.CSTNode):
        """Add a node to be inserted at module level."""
        self.module_insertions.append(node)