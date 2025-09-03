#!/usr/bin/env python3
"""
Simple test to verify libcst backend works
"""

import sys
import os

# Add paconvert to path
sys.path.insert(0, os.path.dirname(__file__))

def test_simple_conversion():
    """Test simple conversion"""
    print("Testing simple conversion...")
    
    # Simple test code
    code = "import torch\nx = torch.tensor([1, 2, 3])"
    
    try:
        from paconvert.converter import Converter
        
        # Create converter with libcst backend
        converter = Converter(backend="libcst", log_dir="disable")
        
        # Create temporary files
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            input_file = f.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            output_file = f.name
        
        try:
            # Run conversion
            success_count, failed_count = converter.run(input_file, output_file)
            
            # Read result
            with open(output_file, 'r') as f:
                result = f.read()
            
            print("Input:")
            print(code)
            print("\nOutput:")
            print(result)
            print(f"\nSuccess: {success_count}, Failed: {failed_count}")
            
            # Check if conversion worked
            if "paddle" in result:
                print("✓ Conversion successful!")
                return True
            else:
                print("✗ No conversion detected")
                return False
                
        finally:
            # Clean up
            os.unlink(input_file)
            os.unlink(output_file)
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_simple_conversion()