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

import unittest
import logging
from unittest.mock import Mock, patch

from paconvert.backend.manager import BackendManager, BACKEND_ERRORS
from paconvert.backend.base import BaseBackend


class MockBackend(BaseBackend):
    """Mock backend for testing."""
    
    def __init__(self, available=True):
        super().__init__()
        self._available = available
    
    def parse_code(self, code: str):
        return f"parsed_{code}"
    
    def generate_code(self, tree):
        return f"generated_{tree}"
    
    def create_transformers(self, tree, file, imports_map, logger, all_api_map=None, unsupport_api_map=None):
        return ["transformer1", "transformer2"]
    
    def is_available(self):
        return self._available


class TestBackendManager(unittest.TestCase):
    """Test cases for BackendManager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.logger = logging.getLogger("test")
        self.logger.setLevel(logging.DEBUG)
    
    def test_valid_backends_list(self):
        """Test that valid backends list is correct."""
        valid_backends = BackendManager.get_valid_backends()
        self.assertEqual(valid_backends, ["astor", "libcst"])
    
    def test_validate_backend_type(self):
        """Test backend type validation."""
        self.assertTrue(BackendManager.validate_backend_type("astor"))
        self.assertTrue(BackendManager.validate_backend_type("libcst"))
        self.assertFalse(BackendManager.validate_backend_type("invalid"))
        self.assertFalse(BackendManager.validate_backend_type(""))
    
    def test_invalid_backend_raises_error(self):
        """Test that invalid backend type raises ValueError."""
        with self.assertRaises(ValueError) as context:
            BackendManager("invalid_backend", self.logger)
        
        self.assertIn("Invalid backend 'invalid_backend'", str(context.exception))
    
    def test_default_backend(self):
        """Test that default backend is astor."""
        self.assertEqual(BackendManager.DEFAULT_BACKEND, "astor")
    
    @patch('paconvert.backend.manager.BackendManager._create_backend')
    def test_backend_creation_success(self, mock_create):
        """Test successful backend creation."""
        mock_backend = MockBackend(available=True)
        mock_create.return_value = mock_backend
        
        manager = BackendManager("astor", self.logger)
        
        self.assertEqual(manager.backend_type, "astor")
        self.assertEqual(manager.requested_backend, "astor")
        self.assertFalse(manager.is_fallback_used())
        self.assertEqual(manager.get_backend_name(), "mockbackend")
    
    @patch('paconvert.backend.manager.BackendManager._create_backend')
    def test_backend_unavailable_fallback(self, mock_create):
        """Test fallback when backend is unavailable."""
        # First call returns unavailable libcst backend
        # Second call returns available astor backend
        unavailable_backend = MockBackend(available=False)
        available_backend = MockBackend(available=True)
        mock_create.side_effect = [unavailable_backend, available_backend]
        
        with patch.object(self.logger, 'warning') as mock_warning:
            manager = BackendManager("libcst", self.logger)
        
        # Should fall back to astor
        self.assertEqual(manager.backend_type, "astor")
        self.assertEqual(manager.requested_backend, "libcst")
        self.assertTrue(manager.is_fallback_used())
        
        # Should log warning about fallback
        mock_warning.assert_called_once()
        self.assertIn("Falling back to astor", mock_warning.call_args[0][0])
    
    @patch('paconvert.backend.manager.BackendManager._create_backend')
    def test_astor_unavailable_raises_error(self, mock_create):
        """Test that unavailable astor backend raises RuntimeError."""
        unavailable_backend = MockBackend(available=False)
        mock_create.return_value = unavailable_backend
        
        with self.assertRaises(RuntimeError) as context:
            BackendManager("astor", self.logger)
        
        self.assertIn("astor backend is not available", str(context.exception))
    
    @patch('paconvert.backend.manager.BackendManager._create_backend')
    def test_backend_methods_delegation(self, mock_create):
        """Test that manager methods delegate to backend correctly."""
        mock_backend = MockBackend(available=True)
        mock_create.return_value = mock_backend
        
        manager = BackendManager("astor", self.logger)
        
        # Test parse_code delegation
        result = manager.parse_code("test_code")
        self.assertEqual(result, "parsed_test_code")
        
        # Test generate_code delegation
        result = manager.generate_code("test_tree")
        self.assertEqual(result, "generated_test_tree")
        
        # Test create_transformers delegation
        result = manager.create_transformers(
            "tree", "file", {}, self.logger, {}, {}
        )
        self.assertEqual(result, ["transformer1", "transformer2"])


if __name__ == "__main__":
    unittest.main()