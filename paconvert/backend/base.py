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

from abc import ABC, abstractmethod
from typing import Any, List, Union
import logging


class BaseBackend(ABC):
    """Abstract base class for AST backends."""
    
    def __init__(self):
        self.backend_name = self.__class__.__name__.replace("Backend", "").lower()
    
    @abstractmethod
    def parse_code(self, code: str) -> Any:
        """Parse source code into AST representation.
        
        Args:
            code: Source code string to parse
            
        Returns:
            AST representation (ast.AST for astor, libcst.Module for libcst)
        """
        pass
    
    @abstractmethod
    def generate_code(self, tree: Any) -> str:
        """Generate source code from AST representation.
        
        Args:
            tree: AST representation to convert back to code
            
        Returns:
            Generated source code string
        """
        pass
    
    @abstractmethod
    def create_transformers(self, tree: Any, file: str, imports_map: dict, 
                          logger: logging.Logger, all_api_map: dict = None, 
                          unsupport_api_map: dict = None) -> List[Any]:
        """Create appropriate transformers for this backend.
        
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
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available (dependencies installed).
        
        Returns:
            True if backend can be used, False otherwise
        """
        pass
    
    def get_backend_name(self) -> str:
        """Get the name of this backend.
        
        Returns:
            Backend name string
        """
        return self.backend_name