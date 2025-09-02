#!/usr/bin/env python3
# Simple verification script

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

try:
    from paconvert.backend import BackendManager
    print("✓ Backend module imported successfully")
    
    # Test backend manager
    manager = BackendManager("astor")
    backend = manager.get_backend()
    print("✓ Astor backend created successfully")
    
    # Test libcst backend (if available)
    try:
        manager_libcst = BackendManager("libcst")
        backend_libcst = manager_libcst.get_backend()
        print("✓ Libcst backend created successfully")
    except ImportError:
        print("⚠ Libcst backend not available (libcst not installed)")
    
    # Test converter with backend
    from paconvert.converter import Converter
    converter = Converter(log_dir="disable", backend="astor")
    print("✓ Converter with astor backend created successfully")
    
    print("\n✓ Backend integration verification completed successfully!")
    
except Exception as e:
    print(f"✗ Verification failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)