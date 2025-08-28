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
import sys
from unittest.mock import Mock, patch
from io import StringIO

from paconvert.backend.validation import (
    validate_backend_availability,
    handle_backend_validation_error,
    create_backend_manager_safely
)


class TestBackendValidation(unittest.TestCase):
    """Test cases for backend validation functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.logger = logging.getLogger("test")
        self.logger.setLevel(logging.DEBUG)
    
    def test_validate_backend_availability_valid_astor(self):
        """Test validation of valid astor backend."""
        is_valid, error_msg = validate_backend_availability("astor", self.logger)
        self.assertTrue(is_valid)
        self.assertEqual(error_msg, "")
    
    def test_validate_backend_availability_invalid_backend(self):
        """Test validation of invalid backend."""
        is_valid, error_msg = validate_backend_availability("invalid", self.logger)
        self.assertFalse(is_valid)
        self.assertIn("Invalid backend 'invalid'", error_msg)
        self.assertIn("Valid options:", error_msg)
    
    @patch('paconvert.backend.validation.libcst', create=True)
    def test_validate_backend_availability_libcst_available(self, mock_libcst):
        """Test validation when libcst is available."""
        is_valid, error_msg = validate_backend_availability("libcst", self.logger)
        self.assertTrue(is_valid)
        self.assertEqual(error_msg, "")
    
    def test_validate_backend_availability_libcst_unavailable(self):
        """Test validation when libcst is not available."""
        # libcst is not installed in test environment
        with patch('builtins.__import__', side_effect=ImportError):
            is_valid, error_msg = validate_backend_availability("libcst", self.logger)
            self.assertFalse(is_valid)
            self.assertIn("libcst is not installed", error_msg)
    
    def test_handle_backend_validation_error_general(self):
        """Test error handling for general backend errors."""
        with patch('sys.stderr', new_callable=StringIO) as mock_stderr:
            with patch('sys.exit') as mock_exit:
                handle_backend_validation_error("invalid", "Test error", self.logger)
                
                # Check that error was printed to stderr
                stderr_output = mock_stderr.getvalue()
                self.assertIn("Error: Test error", stderr_output)
                
                # Check that sys.exit was called
                mock_exit.assert_called_once_with(1)
    
    def test_handle_backend_validation_error_libcst(self):
        """Test error handling for libcst-specific errors."""
        with patch('sys.stderr', new_callable=StringIO) as mock_stderr:
            with patch('sys.exit') as mock_exit:
                handle_backend_validation_error("libcst", "libcst not installed", self.logger)
                
                # Check that libcst-specific help was printed
                stderr_output = mock_stderr.getvalue()
                self.assertIn("pip install libcst", stderr_output)
                self.assertIn("use the default astor backend", stderr_output)
                
                # Check that sys.exit was called
                mock_exit.assert_called_once_with(1)
    
    @patch('paconvert.backend.validation.BackendManager')
    @patch('paconvert.backend.validation.validate_backend_availability')
    def test_create_backend_manager_safely_success(self, mock_validate, mock_manager_class):
        """Test successful backend manager creation."""
        # Mock validation success
        mock_validate.return_value = (True, "")
        
        # Mock backend manager
        mock_manager = Mock()
        mock_manager.is_fallback_used.return_value = False
        mock_manager_class.return_value = mock_manager
        
        result = create_backend_manager_safely("astor", self.logger)
        
        self.assertEqual(result, mock_manager)
        mock_validate.assert_called_once_with("astor", self.logger)
        mock_manager_class.assert_called_once_with("astor", self.logger)
    
    @patch('paconvert.backend.validation.handle_backend_validation_error')
    @patch('paconvert.backend.validation.validate_backend_availability')
    def test_create_backend_manager_safely_validation_failure(self, mock_validate, mock_handle_error):
        """Test backend manager creation with validation failure."""
        # Mock validation failure
        mock_validate.return_value = (False, "Test error")
        
        create_backend_manager_safely("invalid", self.logger)
        
        mock_validate.assert_called_once_with("invalid", self.logger)
        mock_handle_error.assert_called_once_with("invalid", "Test error", self.logger)
    
    @patch('paconvert.backend.validation.handle_backend_validation_error')
    @patch('paconvert.backend.validation.BackendManager')
    @patch('paconvert.backend.validation.validate_backend_availability')
    def test_create_backend_manager_safely_creation_failure(self, mock_validate, mock_manager_class, mock_handle_error):
        """Test backend manager creation with manager creation failure."""
        # Mock validation success
        mock_validate.return_value = (True, "")
        
        # Mock manager creation failure
        mock_manager_class.side_effect = Exception("Creation failed")
        
        create_backend_manager_safely("astor", self.logger)
        
        mock_validate.assert_called_once_with("astor", self.logger)
        mock_handle_error.assert_called_once_with("astor", "Failed to create backend manager: Creation failed", self.logger)
    
    @patch('paconvert.backend.validation.BackendManager')
    @patch('paconvert.backend.validation.validate_backend_availability')
    def test_create_backend_manager_safely_with_fallback(self, mock_validate, mock_manager_class):
        """Test backend manager creation with fallback warning."""
        # Mock validation success
        mock_validate.return_value = (True, "")
        
        # Mock backend manager with fallback
        mock_manager = Mock()
        mock_manager.is_fallback_used.return_value = True
        mock_manager.get_requested_backend.return_value = "libcst"
        mock_manager.get_backend_name.return_value = "astor"
        mock_manager_class.return_value = mock_manager
        
        with patch.object(self.logger, 'warning') as mock_warning:
            result = create_backend_manager_safely("libcst", self.logger)
        
        self.assertEqual(result, mock_manager)
        mock_warning.assert_called_once()
        self.assertIn("Requested backend 'libcst'", mock_warning.call_args[0][0])
        self.assertIn("using 'astor' instead", mock_warning.call_args[0][0])


if __name__ == "__main__":
    unittest.main()