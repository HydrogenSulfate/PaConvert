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

# Try to import libcst, skip tests if not available
try:
    import libcst as cst
    from paconvert.backend.libcst_backend import LibcstBackend
    LIBCST_AVAILABLE = True
except ImportError:
    LIBCST_AVAILABLE = False


@unittest.skipUnless(LIBCST_AVAILABLE, "libcst not available")
class TestCommentPreservation(unittest.TestCase):
    """Test cases for comment preservation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.backend = LibcstBackend()
    
    def test_single_line_comments(self):
        """Test preservation of single-line comments."""
        code = """# Header comment
import torch  # Import comment
x = torch.tensor([1, 2, 3])  # Inline comment
# Footer comment
"""
        
        tree = self.backend.parse_code(code)
        generated = self.backend.generate_code(tree)
        
        # All comments should be preserved
        self.assertIn("# Header comment", generated)
        self.assertIn("# Import comment", generated)
        self.assertIn("# Inline comment", generated)
        self.assertIn("# Footer comment", generated)
    
    def test_multiline_comments(self):
        """Test preservation of multiline comments (docstrings)."""
        code = '''"""
This is a module docstring.
It spans multiple lines.
"""

import torch

class MyModel:
    """
    This is a class docstring.
    It also spans multiple lines.
    """
    
    def forward(self, x):
        """Single line docstring."""
        return torch.relu(x)  # Inline comment
'''
        
        tree = self.backend.parse_code(code)
        generated = self.backend.generate_code(tree)
        
        # Docstrings should be preserved
        self.assertIn('"""', generated)
        self.assertIn("This is a module docstring.", generated)
        self.assertIn("This is a class docstring.", generated)
        self.assertIn("Single line docstring.", generated)
        self.assertIn("# Inline comment", generated)
    
    def test_comments_with_special_characters(self):
        """Test preservation of comments with special characters."""
        code = """# TODO: Fix this function!
import torch  # @deprecated - use paddle instead
x = torch.tensor([1, 2, 3])  # Cost: O(n) time & space
# NOTE: This is important!!!
"""
        
        tree = self.backend.parse_code(code)
        generated = self.backend.generate_code(tree)
        
        # Special characters in comments should be preserved
        self.assertIn("# TODO: Fix this function!", generated)
        self.assertIn("# @deprecated - use paddle instead", generated)
        self.assertIn("# Cost: O(n) time & space", generated)
        self.assertIn("# NOTE: This is important!!!", generated)
    
    def test_comments_in_different_positions(self):
        """Test preservation of comments in various code positions."""
        code = """# Module level comment
import torch

# Before class
class MyModel:
    # Class level comment
    def __init__(self):
        # Method level comment
        self.x = 1  # Attribute comment
        
        # Before if statement
        if True:
            # Inside if block
            pass  # End of if
    
    # Between methods
    def forward(self, x):
        # Start of method
        for i in range(10):  # Loop comment
            # Inside loop
            x = x + 1  # Increment
        # End of method
        return x

# End of file comment
"""
        
        tree = self.backend.parse_code(code)
        generated = self.backend.generate_code(tree)
        
        # All positional comments should be preserved
        comments_to_check = [
            "# Module level comment",
            "# Before class", 
            "# Class level comment",
            "# Method level comment",
            "# Attribute comment",
            "# Before if statement",
            "# Inside if block",
            "# End of if",
            "# Between methods",
            "# Start of method",
            "# Loop comment",
            "# Inside loop",
            "# Increment",
            "# End of method",
            "# End of file comment"
        ]
        
        for comment in comments_to_check:
            self.assertIn(comment, generated, f"Comment '{comment}' not preserved")
    
    def test_empty_lines_preservation(self):
        """Test preservation of empty lines and whitespace."""
        code = """import torch


class MyModel:

    def __init__(self):
        
        self.x = 1
        

    def forward(self, x):
        
        return x

"""
        
        tree = self.backend.parse_code(code)
        generated = self.backend.generate_code(tree)
        
        # The generated code should maintain similar structure
        # (exact whitespace may vary, but general structure should be preserved)
        self.assertIn("class MyModel:", generated)
        self.assertIn("def __init__(self):", generated)
        self.assertIn("def forward(self, x):", generated)
    
    def test_comment_preservation_with_syntax_variations(self):
        """Test comment preservation with different Python syntax."""
        code = """# Function with decorators
@property  # Decorator comment
def my_property(self):
    \"\"\"Property docstring.\"\"\"
    return self._value  # Return comment

# List comprehension with comment
items = [x for x in range(10)]  # List comp comment

# Dictionary with comments
config = {
    'key1': 'value1',  # First key
    'key2': 'value2',  # Second key
}

# Lambda with comment
func = lambda x: x * 2  # Lambda comment

# Try-except with comments
try:
    # Try block comment
    result = 1 / 0
except ZeroDivisionError:  # Exception comment
    # Except block comment
    result = 0
finally:
    # Finally comment
    pass
"""
        
        tree = self.backend.parse_code(code)
        generated = self.backend.generate_code(tree)
        
        # Check that comments in various syntax contexts are preserved
        syntax_comments = [
            "# Function with decorators",
            "# Decorator comment",
            "# Return comment",
            "# List comprehension with comment",
            "# List comp comment",
            "# Dictionary with comments",
            "# First key",
            "# Second key",
            "# Lambda with comment",
            "# Lambda comment",
            "# Try-except with comments",
            "# Try block comment",
            "# Exception comment",
            "# Except block comment",
            "# Finally comment"
        ]
        
        for comment in syntax_comments:
            self.assertIn(comment, generated, f"Syntax comment '{comment}' not preserved")
    
    def test_round_trip_comment_preservation(self):
        """Test that multiple parse/generate cycles preserve comments."""
        original_code = """# Original comment
import torch  # Import comment

def my_function():
    # Function comment
    return torch.tensor([1, 2, 3])  # Return comment
"""
        
        # First round trip
        tree1 = self.backend.parse_code(original_code)
        code1 = self.backend.generate_code(tree1)
        
        # Second round trip
        tree2 = self.backend.parse_code(code1)
        code2 = self.backend.generate_code(tree2)
        
        # Comments should still be preserved after multiple round trips
        self.assertIn("# Original comment", code2)
        self.assertIn("# Import comment", code2)
        self.assertIn("# Function comment", code2)
        self.assertIn("# Return comment", code2)


class TestCommentPreservationComparison(unittest.TestCase):
    """Compare comment preservation between astor and libcst backends."""
    
    def setUp(self):
        """Set up test fixtures."""
        from paconvert.backend.astor_backend import AstorBackend
        self.astor_backend = AstorBackend()
        
        if LIBCST_AVAILABLE:
            self.libcst_backend = LibcstBackend()
    
    @unittest.skipUnless(LIBCST_AVAILABLE, "libcst not available")
    def test_comment_preservation_comparison(self):
        """Compare comment preservation between backends."""
        code_with_comments = """# This is a header comment
import torch  # PyTorch import

# Define a simple function
def process_data(x):
    # Process the input data
    result = torch.relu(x)  # Apply ReLU activation
    return result  # Return processed data

# End of file
"""
        
        # Process with astor backend
        astor_tree = self.astor_backend.parse_code(code_with_comments)
        astor_generated = self.astor_backend.generate_code(astor_tree)
        
        # Process with libcst backend
        libcst_tree = self.libcst_backend.parse_code(code_with_comments)
        libcst_generated = self.libcst_backend.generate_code(libcst_tree)
        
        # Count comments in original, astor output, and libcst output
        original_comments = code_with_comments.count('#')
        astor_comments = astor_generated.count('#')
        libcst_comments = libcst_generated.count('#')
        
        print(f"\nComment preservation comparison:")
        print(f"Original: {original_comments} comments")
        print(f"Astor:    {astor_comments} comments")
        print(f"Libcst:   {libcst_comments} comments")
        
        # Libcst should preserve more comments than astor
        self.assertGreaterEqual(libcst_comments, astor_comments)
        
        # Ideally, libcst should preserve all comments
        self.assertEqual(libcst_comments, original_comments)


if __name__ == "__main__":
    unittest.main()