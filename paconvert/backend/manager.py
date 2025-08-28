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
from typing import Any, List, Union

from .base import BaseBackend


# Error message templates
BACKEND_ERRORS = {
    "invalid_backend": "Invalid backend '{backend}'. Valid options: astor, libcst",
    "libcst_not_installed": "libcst is not installed. Install with: pip install libcst",
    "astor_not_available": "astor backend is not available. This should not happen.",
    "backend_creation_failed": "Failed to create {backend} backend: {error}",
    "fallback_to_astor": "Falling back to astor backend due to: {reason}",
}


class BackendManager:
    """Manages AST backend selection and operations."""
    
    VALID_BACKENDS = ["astor", "libcst"]
    DEFAULT_BACKEND = "astor"
    
    def __init__(self, backend_type: str = DEFAULT_BACKEND, logger: logging.Logger = None):
        """Initialize backend manager.
        
        Args:
            backend_type: Type of backend to use ("astor" or "libcst")
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.requested_backend = backend_type
        self.backend_type = backend_type
        self.backend = None
        
        # Validate and create backend
        self._validate_and_create_backend()
    
    def _validate_and_create_backend(self):
        """Validate backend type and create backend instance."""
        # Validate backend type
        if self.backend_type not in self.VALID_BACKENDS:
            error_msg = BACKEND_ERRORS["invalid_backend"].format(backend=self.backend_type)
            raise ValueError(error_msg)
        
        # Try to create requested backend
        try:
            self.backend = self._create_backend(self.backend_type)
            if not self.backend.is_available():
                self._handle_backend_unavailable()
        except Exception as e:
            error_msg = BACKEND_ERRORS["backend_creation_failed"].format(
                backend=self.backend_type, error=str(e)
            )
            self.logger.error(error_msg)
            self._handle_backend_unavailable()
    
    def _handle_backend_unavailable(self):
        """Handle case when requested backend is unavailable."""
        if self.backend_type == "libcst":
            # Fall back to astor
            fallback_msg = BACKEND_ERRORS["fallback_to_astor"].format(
                reason="libcst not available"
            )
            self.logger.warning(fallback_msg)
            self.backend_type = "astor"
            self.backend = self._create_backend("astor")
            
            if not self.backend.is_available():
                raise RuntimeError(BACKEND_ERRORS["astor_not_available"])
        else:
            raise RuntimeError(BACKEND_ERRORS["astor_not_available"])
    
    def _create_backend(self, backend_type: str) -> BaseBackend:
        """Create backend instance.
        
        Args:
            backend_type: Type of backend to create
            
        Returns:
            Backend instance
        """
        if backend_type == "astor":
            from .astor_backend import AstorBackend
            return AstorBackend()
        elif backend_type == "libcst":
            from .libcst_backend import LibcstBackend
            return LibcstBackend()
        else:
            raise ValueError(f"Unknown backend type: {backend_type}")
    
    def parse_code(self, code: str) -> Any:
        """Parse source code using the selected backend.
        
        Args:
            code: Source code to parse
            
        Returns:
            AST representation
        """
        return self.backend.parse_code(code)
    
    def generate_code(self, tree: Any) -> str:
        """Generate source code using the selected backend.
        
        Args:
            tree: AST representation
            
        Returns:
            Generated source code
        """
        return self.backend.generate_code(tree)
    
    def create_transformers(self, tree: Any, file: str, imports_map: dict,
                          logger: logging.Logger, all_api_map: dict = None,
                          unsupport_api_map: dict = None) -> List[Any]:
        """Create transformers using the selected backend.
        
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
        return self.backend.create_transformers(
            tree, file, imports_map, logger, all_api_map, unsupport_api_map
        )
    
    def get_backend_name(self) -> str:
        """Get the name of the active backend.
        
        Returns:
            Backend name
        """
        return self.backend.get_backend_name()
    
    def get_requested_backend(self) -> str:
        """Get the originally requested backend name.
        
        Returns:
            Requested backend name
        """
        return self.requested_backend
    
    def is_fallback_used(self) -> bool:
        """Check if fallback backend is being used.
        
        Returns:
            True if fallback is used, False otherwise
        """
        return self.requested_backend != self.backend_type
    
    @classmethod
    def validate_backend_type(cls, backend_type: str) -> bool:
        """Validate if backend type is supported.
        
        Args:
            backend_type: Backend type to validate
            
        Returns:
            True if valid, False otherwise
        """
        return backend_type in cls.VALID_BACKENDS
    
    @classmethod
    def get_valid_backends(cls) -> List[str]:
        """Get list of valid backend types.
        
        Returns:
            List of valid backend names
        """
        return cls.VALID_BACKENDS.copy()