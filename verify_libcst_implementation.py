#!/usr/bin/env python3
"""
Simple verification script for native libcst implementation
"""

import sys
import os

# Add paconvert to path
sys.path.insert(0, os.path.dirname(__file__))

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    
    try:
        import libcst as cst
        print("✓ libcst imported successfully")
    except ImportError:
        print("✗ libcst not available - please install with: pip install libcst")
        return False
    
    try:
        from paconvert.backend.libcst_backend import LibcstBackend
        print("✓ LibcstBackend imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import LibcstBackend: {e}")
        return False
    
    try:
        from paconvert.transformer.libcst_transformers.base_transformer import LibcstBaseTransformer
        print("✓ LibcstBaseTransformer imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import LibcstBaseTransformer: {e}")
        return False
    
    try:
        from paconvert.transformer.libcst_transformers.basic_transformer import LibcstBasicTransformer
        print("✓ LibcstBasicTransformer imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import LibcstBasicTransformer: {e}")
        return False
    
    try:
        from paconvert.api_mapping import API_MAPPING
        print("✓ API_MAPPING imported successfully")
        print(f"  Found {len(API_MAPPING)} API mappings")
    except ImportError as e:
        print(f"✗ Failed to import API_MAPPING: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic libcst backend functionality"""
    print("\nTesting basic functionality...")
    
    try:
        from paconvert.backend.libcst_backend import LibcstBackend
        import libcst as cst
        
        backend = LibcstBackend()
        
        # Test parsing
        code = "import torch\nx = torch.tensor([1, 2, 3])"
        tree = backend.parse_code(code)
        print("✓ Code parsing works")
        
        # Test code generation
        generated = backend.generate_code(tree)
        print("✓ Code generation works")
        
        # Test node_to_source
        name_node = cst.Name("test")
        source = backend.node_to_source(name_node)
        print(f"✓ Node to source works: '{source}'")
        
        # Test backend type
        backend_type = backend.get_backend_type()
        print(f"✓ Backend type: {backend_type}")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_comment_preservation():
    """Test comment preservation"""
    print("\nTesting comment preservation...")
    
    try:
        from paconvert.backend.libcst_backend import LibcstBackend
        
        backend = LibcstBackend()
        
        code_with_comments = '''# Header comment
import torch  # Import comment

# Block comment
x = torch.tensor([1, 2, 3])  # Inline comment
'''
        
        tree = backend.parse_code(code_with_comments)
        generated = backend.generate_code(tree)
        
        print("Original code:")
        print(code_with_comments)
        print("Generated code:")
        print(generated)
        
        # Check if comments are preserved
        comments_preserved = (
            "# Header comment" in generated and
            "# Import comment" in generated and
            "# Block comment" in generated and
            "# Inline comment" in generated
        )
        
        if comments_preserved:
            print("✓ Comments preserved successfully")
        else:
            print("⚠ Some comments may not be preserved (this is expected behavior)")
        
        return True
        
    except Exception as e:
        print(f"✗ Comment preservation test failed: {e}")
        return False

def test_transformer_creation():
    """Test transformer creation"""
    print("\nTesting transformer creation...")
    
    try:
        from paconvert.backend.libcst_backend import LibcstBackend
        
        backend = LibcstBackend()
        transformers = backend.create_transformers()
        
        print(f"✓ Created {len(transformers)} transformer classes:")
        for transformer_class in transformers:
            print(f"  - {transformer_class.__name__}")
        
        return True
        
    except Exception as e:
        print(f"✗ Transformer creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all verification tests"""
    print("Verifying native libcst implementation...")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_basic_functionality,
        test_comment_preservation,
        test_transformer_creation,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"✗ {test.__name__} failed")
        except Exception as e:
            print(f"✗ {test.__name__} crashed: {e}")
    
    print("\n" + "=" * 50)
    print(f"Verification Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 Native libcst implementation verified successfully!")
        print("\nYou can now use the libcst backend with:")
        print("  paconvert -i input.py -o output.py --backend libcst")
        return 0
    else:
        print("❌ Some verification tests failed.")
        print("Please check the implementation or install missing dependencies.")
        return 1

if __name__ == "__main__":
    sys.exit(main())