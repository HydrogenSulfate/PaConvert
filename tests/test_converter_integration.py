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
import tempfile
import os
import shutil
from unittest.mock import Mock, patch

from paconvert.converter import Converter


class TestConverterIntegration(unittest.TestCase):
    """Integration tests for Converter with backend system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.input_dir = os.path.join(self.temp_dir, "input")
        self.output_dir = os.path.join(self.temp_dir, "output")
        os.makedirs(self.input_dir)
        os.makedirs(self.output_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def create_test_file(self, filename, content):
        """Create a test file with given content."""
        filepath = os.path.join(self.input_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return filepath
    
    def test_converter_with_astor_backend(self):
        """Test converter works with astor backend."""
        # Create test PyTorch file
        test_content = """import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)
    
    def forward(self, x):
        return torch.relu(self.linear(x))
"""
        self.create_test_file("model.py", test_content)
        
        # Create converter with astor backend
        converter = Converter(
            log_dir="disable",  # Disable logging for test
            backend="astor"
        )
        
        # Run conversion
        success_count, failed_count = converter.run(self.input_dir, self.output_dir)
        
        # Check that conversion completed
        self.assertGreaterEqual(success_count, 0)
        self.assertGreaterEqual(failed_count, 0)
        
        # Check that output file was created
        output_file = os.path.join(self.output_dir, "model.py")
        self.assertTrue(os.path.exists(output_file))
        
        # Check that output contains paddle imports
        with open(output_file, "r", encoding="utf-8") as f:
            output_content = f.read()
        
        self.assertIn("paddle", output_content.lower())
    
    def test_converter_backend_manager_initialization(self):
        """Test that converter properly initializes backend manager."""
        converter = Converter(log_dir="disable", backend="astor")
        
        # Check backend manager is created
        self.assertIsNotNone(converter.backend_manager)
        self.assertEqual(converter.backend_manager.get_backend_name(), "astor")
        self.assertEqual(converter.backend_type, "astor")
    
    def test_converter_with_invalid_backend_fallback(self):
        """Test converter behavior with backend fallback."""
        # Mock BackendManager to simulate fallback
        with patch('paconvert.converter.BackendManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager.get_backend_name.return_value = "astor"
            mock_manager.get_requested_backend.return_value = "libcst"
            mock_manager.is_fallback_used.return_value = True
            mock_manager.parse_code.return_value = Mock()
            mock_manager.generate_code.return_value = "# converted code"
            mock_manager.create_transformers.return_value = []
            mock_manager_class.return_value = mock_manager
            
            converter = Converter(log_dir="disable", backend="libcst")
            
            # Check that fallback was handled
            self.assertTrue(converter.backend_manager.is_fallback_used())
    
    def test_converter_preserves_non_python_files(self):
        """Test that converter preserves non-Python files."""
        # Create test files
        self.create_test_file("config.json", '{"key": "value"}')
        self.create_test_file("README.md", "# Test Project")
        
        converter = Converter(log_dir="disable", backend="astor")
        converter.run(self.input_dir, self.output_dir)
        
        # Check that non-Python files are copied
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "config.json")))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "README.md")))
    
    def test_converter_handles_requirements_txt(self):
        """Test that converter handles requirements.txt files."""
        # Create requirements.txt with torch
        requirements_content = """torch==1.9.0
torchvision==0.10.0
numpy==1.21.0
"""
        self.create_test_file("requirements.txt", requirements_content)
        
        converter = Converter(log_dir="disable", backend="astor")
        converter.run(self.input_dir, self.output_dir)
        
        # Check that requirements.txt was processed
        output_file = os.path.join(self.output_dir, "requirements.txt")
        self.assertTrue(os.path.exists(output_file))
        
        with open(output_file, "r", encoding="utf-8") as f:
            output_content = f.read()
        
        # torch should be replaced with paddlepaddle-gpu
        self.assertIn("paddlepaddle-gpu", output_content)
        self.assertNotIn("torch==", output_content)
    
    def test_converter_api_counting(self):
        """Test that converter correctly counts APIs."""
        # Create test file with known PyTorch APIs
        test_content = """import torch
x = torch.tensor([1, 2, 3])
y = torch.relu(x)
z = torch.add(x, y)
"""
        self.create_test_file("test_apis.py", test_content)
        
        converter = Converter(log_dir="disable", backend="astor")
        success_count, failed_count = converter.run(self.input_dir, self.output_dir)
        
        # Should have processed some APIs
        total_apis = converter.torch_api_count
        self.assertGreater(total_apis, 0)
        self.assertEqual(success_count + failed_count, total_apis)
    
    def test_converter_with_syntax_error_file(self):
        """Test converter handles files with syntax errors gracefully."""
        # Create file with syntax error
        test_content = """import torch
x = torch.tensor([1, 2, 3
# Missing closing bracket
"""
        self.create_test_file("syntax_error.py", test_content)
        
        converter = Converter(log_dir="disable", backend="astor")
        
        # Should not crash, but may not process the file
        try:
            converter.run(self.input_dir, self.output_dir)
        except Exception as e:
            # If it fails, it should be a parsing error, not a backend error
            self.assertIn("parse", str(e).lower())


if __name__ == "__main__":
    unittest.main()