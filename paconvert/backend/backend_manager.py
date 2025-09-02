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
from typing import Optional

from .base_backend import BaseBackend
from .astor_backend import AstorBackend
from .libcst_backend import LibcstBackend


class BackendManager:
    """Manager class for handling different code transformation backends."""
    
    _backends = {
        "astor": AstorBackend,
        "libcst": LibcstBackend,
    }
    
    def __init__(self, backend_name: str = "astor", logger: Optional[logging.Logger] = None):
        self.backend_name = backend_name
        self.logger = logger or logging.getLogger(__name__)
        self._backend = None
        
    def get_backend(self) -> BaseBackend:
        """Get the configured backend instance."""
        if self._backend is None:
            self._backend = self._create_backend()
        return self._backend
    
    def _create_backend(self) -> BaseBackend:
        """Create and return the appropriate backend instance."""
        if self.backend_name not in self._backends:
            available_backends = ", ".join(self._backends.keys())
            raise ValueError(
                f"Unknown backend '{self.backend_name}'. "
                f"Available backends: {available_backends}"
            )
        
        backend_class = self._backends[self.backend_name]
        
        try:
            backend = backend_class()
            if self.logger:
                self.logger.info(f"Using {self.backend_name} backend for code transformation")
            return backend
        except ImportError as e:
            if self.backend_name == "libcst":
                if self.logger:
                    self.logger.warning(
                        f"libcst backend requested but not available: {e}. "
                        "Falling back to astor backend. "
                        "Install libcst with: pip install libcst"
                    )
                # Fall back to astor backend
                self.backend_name = "astor"
                return self._backends["astor"]()
            else:
                raise
    
    @classmethod
    def get_available_backends(cls):
        """Get list of available backend names."""
        return list(cls._backends.keys())