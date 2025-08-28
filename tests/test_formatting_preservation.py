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

# Try to import libcst, skip tests if not available
try:
    import libcst as cst
    from paconvert.backend.libcst_backend import LibcstBackend
    LIBCST_AVAILABLE = True
except ImportError:
    LIBCST_AVAILABLE = False


@unittest.skipUnless(LIBCST_AVAILABLE, "libcst not available")
class TestFormattingPreservation(unittest.TestCase):
    """Test cases for formatting preservation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.backend = LibcstBackend()
    
    def test_indentation_preservation(self):
        """Test preservation of indentation styles."""
        code = """import torch

class MyModel:
    def __init__(self):
        self.layer1 = torch.nn.Linear(10, 5)
        self.layer2 = torch.nn.Linear(5, 1)
    
    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        return x
"""
        
        tree = self.backend.parse_code(code)
        generated = self.backend.generate_code(tree)
        
        # Check that indentation is preserved
        self.assertEqual(generated, code)
        
        # Check specific indentation patterns
        lines = generated.splitlines()
        class_line = next(line for line in lines if "class MyModel:" in line)
        method_line = next(line for line in lines if "def __init__" in line)
        statement_line = next(line for line in lines if "self.layer1" in line)
        
        # Class should not be indented
        self.assertFalse(class_line.startswith(' '))
        
        # Method should be indented with 4 spaces
        self.assertTrue(method_line.startswith('    '))
        
        # Statements should be indented with 8 spaces
        self.assertTrue(statement_line.startswith('        '))
    
    def test_blank_line_preservation(self):
        """Test preservation of blank lines."""
        code = """import torch


class MyModel:

    def __init__(self):

        self.x = 1


    def forward(self, x):

        return x


"""
        
        tree = self.backend.parse_code(code)
        generated = self.backend.generate_code(tree)
        
        # Should preserve the exact formatting including blank lines
        self.assertEqual(generated, code)
    
    def test_spacing_preservation(self):
        """Test preservation of spacing around operators and punctuation."""
        code = """import torch

# Different spacing styles
x=torch.tensor([1,2,3])
y = torch.tensor([1, 2, 3])
z  =  torch.tensor( [ 1 , 2 , 3 ] )

# Function calls with different spacing
result1=torch.add(x,y)
result2 = torch.add( x , y )
result3=torch.add(x, y)
"""
        
        tree = self.backend.parse_code(code)
        generated = self.backend.generate_code(tree)
        
        # Should preserve exact spacing
        self.assertEqual(generated, code)
    
    def test_line_continuation_preservation(self):
        """Test preservation of line continuations."""
        code = """import torch

# Line continuation with backslash
x = torch.tensor([1, 2, 3]) + \\
    torch.tensor([4, 5, 6])

# Implicit line continuation
y = (torch.tensor([1, 2, 3]) +
     torch.tensor([4, 5, 6]))

# Function call with line breaks
result = torch.add(
    torch.tensor([1, 2, 3]),
    torch.tensor([4, 5, 6])
)
"""
        
        tree = self.backend.parse_code(code)
        generated = self.backend.generate_code(tree)
        
        # Should preserve line continuation styles
        self.assertEqual(generated, code)
    
    def test_string_formatting_preservation(self):
        """Test preservation of string formatting styles."""
        code = '''import torch

# Different string quote styles
single_quotes = 'Hello World'
double_quotes = "Hello World"
triple_single = \\'''Multi
line
string\\'''
triple_double = """Another
multi
line string"""

# Raw strings
raw_string = r"Raw string with \\n"
f_string = f"Tensor shape: {torch.tensor([1, 2, 3]).shape}"
'''
        
        tree = self.backend.parse_code(code)
        generated = self.backend.generate_code(tree)
        
        # Should preserve string formatting
        self.assertEqual(generated, code)
    
    def test_complex_formatting_preservation(self):
        """Test preservation of complex formatting scenarios."""
        code = """import torch
import torch.nn as nn


class ComplexModel(nn.Module):
    \"\"\"A complex model with various formatting styles.\"\"\"
    
    def __init__(self,
                 input_size: int = 10,
                 hidden_size: int = 5,
                 output_size: int = 1):
        super().__init__()
        
        # Different formatting styles in same class
        self.layer1=nn.Linear(input_size,hidden_size)
        self.layer2 = nn.Linear( hidden_size , output_size )
        
        self.activation = nn.ReLU()
    
    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        \"\"\"Forward pass with type hints.\"\"\"
        
        # Chain operations with different spacing
        x=self.layer1(x)
        x = self.activation( x )
        x=self.layer2(x)
        
        return x


# Function with complex parameters
def create_model(input_size=10,
                hidden_size=5,
                output_size=1):
    return ComplexModel(
        input_size=input_size,
        hidden_size = hidden_size,
        output_size  =  output_size
    )
"""
        
        tree = self.backend.parse_code(code)
        generated = self.backend.generate_code(tree)
        
        # Should preserve all the complex formatting
        self.assertEqual(generated, code)
    
    def test_nested_structure_formatting(self):
        """Test preservation of formatting in nested structures."""
        code = """import torch

# Nested data structures with various formatting
config = {
    'model': {
        'type': 'linear',
        'params': {
            'input_size':10,
            'output_size' : 1
        }
    },
    'training':{
        'lr':0.01,
        'epochs' : 100
    }
}

# Nested function calls
result = torch.add(
    torch.mul(
        torch.tensor([1,2,3]),
        torch.tensor([2,3,4])
    ),
    torch.tensor([1, 1, 1])
)

# List comprehension with formatting
data=[
    torch.tensor([i,i+1,i+2])
    for i in range(5)
]
"""
        
        tree = self.backend.parse_code(code)
        generated = self.backend.generate_code(tree)
        
        # Should preserve nested formatting
        self.assertEqual(generated, code)


class TestFormattingPreservationComparison(unittest.TestCase):
    """Compare formatting preservation between astor and libcst backends."""
    
    def setUp(self):
        """Set up test fixtures."""
        from paconvert.backend.astor_backend import AstorBackend
        self.astor_backend = AstorBackend()
        
        if LIBCST_AVAILABLE:
            self.libcst_backend = LibcstBackend()
    
    @unittest.skipUnless(LIBCST_AVAILABLE, "libcst not available")
    def test_formatting_preservation_comparison(self):
        """Compare formatting preservation between backends."""
        original_code = """import torch

# Preserve this spacing
x=torch.tensor([1,2,3])
y = torch.tensor( [4, 5, 6] )

# Preserve this indentation
if True:
    result=torch.add(x,y)
"""
        
        # Process with astor backend
        astor_tree = self.astor_backend.parse_code(original_code)
        astor_generated = self.astor_backend.generate_code(astor_tree)
        
        # Process with libcst backend
        libcst_tree = self.libcst_backend.parse_code(original_code)
        libcst_generated = self.libcst_backend.generate_code(libcst_tree)
        
        print(f"\nFormatting preservation comparison:")
        print(f"Original length: {len(original_code)} chars")
        print(f"Astor length:    {len(astor_generated)} chars")
        print(f"Libcst length:   {len(libcst_generated)} chars")
        
        # Libcst should preserve exact formatting
        self.assertEqual(libcst_generated, original_code)
        
        # Astor typically reformats code
        # (This assertion might fail, which demonstrates the difference)
        if astor_generated != original_code:
            print("Astor reformatted the code (expected)")
        else:
            print("Astor preserved formatting (unexpected)")
    
    @unittest.skipUnless(LIBCST_AVAILABLE, "libcst not available")
    def test_whitespace_sensitivity(self):
        """Test sensitivity to whitespace differences."""
        # Code with intentional whitespace variations
        code_variations = [
            "x=torch.tensor([1,2,3])",
            "x = torch.tensor([1, 2, 3])",
            "x  =  torch.tensor( [ 1 , 2 , 3 ] )",
        ]
        
        for code in code_variations:
            with self.subTest(code=code):
                # Libcst should preserve exact formatting
                tree = self.libcst_backend.parse_code(code)
                generated = self.libcst_backend.generate_code(tree)
                self.assertEqual(generated, code)
                
                # Astor typically normalizes formatting
                astor_tree = self.astor_backend.parse_code(code)
                astor_generated = self.astor_backend.generate_code(astor_tree)
                # We don't assert equality here since astor reformats


if __name__ == "__main__":
    unittest.main()