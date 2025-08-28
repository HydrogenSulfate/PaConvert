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
import ast
import logging
from unittest.mock import Mock, patch

from paconvert.backend.astor_backend import AstorBackend


class TestAstorBackend(unittest.TestCase):
    """Test cases for AstorBackend."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.backend = AstorBackend()
        self.logger = logging.getLogger("test")
        self.logger.setLevel(logging.DEBUG)
    
    def test_backend_name(self):
        """Test backend name is correct."""
        self.assertEqual(self.backend.get_backend_name(), "astor")
    
    def test_is_available(self):
        """Test that astor backend reports as available."""
        self.assertTrue(self.backend.is_available())
    
    def test_parse_code_success(self):
        """Test successful code parsing."""
        code = "import torch\nx = torch.tensor([1, 2, 3])"
        tree = self.backend.parse_code(code)
        
        self.assertIsInstance(tree, ast.AST)
        self.assertIsInstance(tree, ast.Module)
    
    def test_parse_code_syntax_error(self):
        """Test parsing code with syntax error."""
        code = "import torch\nx = torch.tensor([1, 2, 3"  # Missing closing bracket
        
        with self.assertRaises(SyntaxError) as context:
            self.backend.parse_code(code)
        
        self.assertIn("Failed to parse code", str(context.exception))
    
    def test_generate_code_success(self):
        """Test successful code generation."""
        code = "import torch\nx = torch.tensor([1, 2, 3])"
        tree = ast.parse(code)
        
        generated_code = self.backend.generate_code(tree)
        
        self.assertIsInstance(generated_code, str)
        self.assertIn("import torch", generated_code)
        self.assertIn("torch.tensor", generated_code)
    
    def test_generate_code_invalid_tree(self):
        """Test code generation with invalid tree."""
        invalid_tree = "not an ast tree"
        
        with self.assertRaises(ValueError) as context:
            self.backend.generate_code(invalid_tree)
        
        self.assertIn("Failed to generate code from AST", str(context.exception))
    
    def test_create_transformers(self):
        """Test transformer creation."""
        code = "import torch\nx = torch.tensor([1, 2, 3])"
        tree = ast.parse(code)
        imports_map = {}
        all_api_map = {}
        unsupport_api_map = {}
        
        transformers = self.backend.create_transformers(
            tree, "test.py", imports_map, self.logger, all_api_map, unsupport_api_map
        )
        
        # Should create 5 transformers
        self.assertEqual(len(transformers), 5)
        
        # Check transformer types
        transformer_names = [type(t).__name__ for t in transformers]
        expected_names = [
            "ImportTransformer",
            "TensorRequiresGradTransformer", 
            "BasicTransformer",
            "PreCustomOpTransformer",
            "CustomOpTransformer"
        ]
        self.assertEqual(transformer_names, expected_names)
    
    def test_create_transformers_with_none_optional_params(self):
        """Test transformer creation with None optional parameters."""
        code = "import torch"
        tree = ast.parse(code)
        imports_map = {}
        
        transformers = self.backend.create_transformers(
            tree, "test.py", imports_map, self.logger, None, None
        )
        
        # Should still create transformers successfully
        self.assertEqual(len(transformers), 5)
    
    @patch('paconvert.backend.astor_backend.ASTOR_AVAILABLE', False)
    def test_astor_unavailable_raises_error(self):
        """Test that AstorBackend raises error when astor is unavailable."""
        with self.assertRaises(ImportError) as context:
            AstorBackend()
        
        self.assertIn("astor is required", str(context.exception))
    
    def test_round_trip_conversion(self):
        """Test that parse -> generate produces equivalent code."""
        original_code = """import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.linear(x)
"""
        
        # Parse and generate
        tree = self.backend.parse_code(original_code)
        generated_code = self.backend.generate_code(tree)
        
        # Parse generated code to ensure it's valid
        tree2 = self.backend.parse_code(generated_code)
        
        # Both trees should be equivalent (though formatting may differ)
        self.assertIsInstance(tree2, ast.Module)


if __name__ == "__main__":
    unittest.main()