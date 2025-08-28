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

# Try to import libcst, skip tests if not available
try:
    import libcst as cst
    from paconvert.backend.libcst_backend import LibcstBackend, LibcstTransformerBridge
    LIBCST_AVAILABLE = True
except ImportError:
    LIBCST_AVAILABLE = False


@unittest.skipUnless(LIBCST_AVAILABLE, "libcst not available")
class TestLibcstBackend(unittest.TestCase):
    """Test cases for LibcstBackend."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.backend = LibcstBackend()
        self.logger = logging.getLogger("test")
        self.logger.setLevel(logging.DEBUG)
    
    def test_backend_name(self):
        """Test backend name is correct."""
        self.assertEqual(self.backend.get_backend_name(), "libcst")
    
    def test_is_available(self):
        """Test that libcst backend reports as available when libcst is installed."""
        self.assertTrue(self.backend.is_available())
    
    def test_parse_code_success(self):
        """Test successful code parsing."""
        code = "import torch\nx = torch.tensor([1, 2, 3])"
        tree = self.backend.parse_code(code)
        
        self.assertIsInstance(tree, cst.Module)
    
    def test_parse_code_with_comments(self):
        """Test parsing code with comments."""
        code = """# This is a comment
import torch  # Import PyTorch
# Another comment
x = torch.tensor([1, 2, 3])  # Create tensor
"""
        tree = self.backend.parse_code(code)
        
        self.assertIsInstance(tree, cst.Module)
        # The original code should be preserved
        self.assertEqual(tree.code, code)
    
    def test_parse_code_syntax_error(self):
        """Test parsing code with syntax error."""
        code = "import torch\nx = torch.tensor([1, 2, 3"  # Missing closing bracket
        
        with self.assertRaises(Exception):  # libcst raises ParserError
            self.backend.parse_code(code)
    
    def test_generate_code_success(self):
        """Test successful code generation."""
        code = "import torch\nx = torch.tensor([1, 2, 3])"
        tree = self.backend.parse_code(code)
        
        generated_code = self.backend.generate_code(tree)
        
        self.assertIsInstance(generated_code, str)
        self.assertEqual(generated_code, code)  # Should preserve exact formatting
    
    def test_generate_code_with_comments(self):
        """Test code generation preserves comments."""
        code = """# This is a comment
import torch  # Import PyTorch
x = torch.tensor([1, 2, 3])  # Create tensor
"""
        tree = self.backend.parse_code(code)
        generated_code = self.backend.generate_code(tree)
        
        # Comments should be preserved
        self.assertIn("# This is a comment", generated_code)
        self.assertIn("# Import PyTorch", generated_code)
        self.assertIn("# Create tensor", generated_code)
    
    def test_create_transformers(self):
        """Test transformer creation."""
        code = "import torch\nx = torch.tensor([1, 2, 3])"
        tree = self.backend.parse_code(code)
        imports_map = {}
        all_api_map = {}
        unsupport_api_map = {}
        
        transformers = self.backend.create_transformers(
            tree, "test.py", imports_map, self.logger, all_api_map, unsupport_api_map
        )
        
        # Should create 5 bridge transformers
        self.assertEqual(len(transformers), 5)
        
        # All should be bridge instances
        for transformer in transformers:
            self.assertIsInstance(transformer, LibcstTransformerBridge)
    
    def test_round_trip_conversion(self):
        """Test that parse -> generate preserves code exactly."""
        original_code = """# Model definition
import torch
import torch.nn as nn

class MyModel(nn.Module):
    \"\"\"A simple model.\"\"\"
    
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)  # Linear layer
    
    def forward(self, x):
        # Apply linear transformation
        return torch.relu(self.linear(x))
"""
        
        # Parse and generate
        tree = self.backend.parse_code(original_code)
        generated_code = self.backend.generate_code(tree)
        
        # Should be identical
        self.assertEqual(generated_code, original_code)


class TestLibcstBackendUnavailable(unittest.TestCase):
    """Test LibcstBackend behavior when libcst is not available."""
    
    @patch('paconvert.backend.libcst_backend.LIBCST_AVAILABLE', False)
    def test_libcst_unavailable_raises_error(self):
        """Test that LibcstBackend raises error when libcst is unavailable."""
        with self.assertRaises(ImportError) as context:
            LibcstBackend()
        
        self.assertIn("libcst is required", str(context.exception))
        self.assertIn("pip install libcst", str(context.exception))


@unittest.skipUnless(LIBCST_AVAILABLE, "libcst not available")
class TestLibcstTransformerBridge(unittest.TestCase):
    """Test cases for LibcstTransformerBridge."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.logger = logging.getLogger("test")
        self.logger.setLevel(logging.DEBUG)
    
    def test_bridge_creation(self):
        """Test bridge transformer creation."""
        # Mock AST transformer class
        mock_transformer_class = Mock()
        
        bridge = LibcstTransformerBridge(mock_transformer_class)
        self.assertEqual(bridge.ast_transformer_class, mock_transformer_class)
    
    def test_bridge_instance_creation(self):
        """Test bridge instance creation."""
        mock_transformer_class = Mock()
        bridge = LibcstTransformerBridge(mock_transformer_class)
        
        code = "import torch"
        tree = cst.parse_module(code)
        
        instance = bridge(tree, "test.py", {}, self.logger, {}, {})
        
        self.assertIsNotNone(instance)
        self.assertEqual(instance.ast_transformer_class, mock_transformer_class)
        self.assertEqual(instance.libcst_tree, tree)
        self.assertEqual(instance.file, "test.py")
    
    @patch('paconvert.backend.libcst_backend.ast')
    @patch('paconvert.backend.libcst_backend.astor')
    def test_bridge_transform(self, mock_astor, mock_ast):
        """Test bridge transformation process."""
        # Mock AST transformer
        mock_transformer_instance = Mock()
        mock_transformer_instance.torch_api_count = 5
        mock_transformer_instance.success_api_count = 4
        
        mock_transformer_class = Mock()
        mock_transformer_class.return_value = mock_transformer_instance
        
        # Mock AST operations
        mock_ast_tree = Mock()
        mock_ast.parse.return_value = mock_ast_tree
        mock_astor.to_source.return_value = "transformed code"
        
        # Create bridge instance
        bridge = LibcstTransformerBridge(mock_transformer_class)
        code = "import torch"
        tree = cst.parse_module(code)
        
        instance = bridge(tree, "test.py", {}, self.logger, {}, {})
        instance.transform()
        
        # Check that AST transformer was called
        mock_transformer_class.assert_called_once()
        mock_transformer_instance.transform.assert_called_once()
        
        # Check that counts were updated
        self.assertEqual(instance.torch_api_count, 5)
        self.assertEqual(instance.success_api_count, 4)


if __name__ == "__main__":
    unittest.main()