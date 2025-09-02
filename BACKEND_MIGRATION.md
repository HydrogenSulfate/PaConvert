# PaConvert Backend Migration Guide

This document describes the migration from astor to libcst backend in PaConvert.

## Overview

PaConvert has been refactored to support multiple backends for code transformation:

- **astor backend** (default): Uses the existing astor library for AST to source conversion
- **libcst backend**: Uses libcst for better comment and formatting preservation

## Usage

### Command Line

You can specify the backend using the `--backend` parameter:

```bash
# Use astor backend (default)
paconvert -i torch_code_dir -o paddle_code_dir

# Explicitly use astor backend
paconvert -i torch_code_dir -o paddle_code_dir --backend astor

# Use libcst backend
paconvert -i torch_code_dir -o paddle_code_dir --backend libcst
```

### Python API

```python
from paconvert.converter import Converter

# Use astor backend (default)
converter = Converter(backend="astor")

# Use libcst backend
converter = Converter(backend="libcst")

converter.run(input_dir, output_dir)
```

## Backend Comparison

| Feature | Astor Backend | Libcst Backend |
|---------|---------------|----------------|
| Comment preservation | ❌ Comments are removed | ✅ Comments are preserved |
| Formatting preservation | ❌ Code is reformatted | ✅ Better formatting preservation |
| Conversion accuracy | ✅ Fully tested | ✅ Equivalent accuracy |
| Performance | ✅ Fast | ✅ Similar performance |
| Dependencies | astor | libcst |

## Installation

### Astor Backend (Default)
```bash
pip install astor
```

### Libcst Backend
```bash
pip install libcst
```

### Both Backends
```bash
pip install astor libcst
```

## Migration Notes

### For Users
- The default backend remains `astor` for backward compatibility
- All existing functionality works exactly as before
- To use libcst features, simply add `--backend libcst` to your command

### For Developers
- The backend abstraction is implemented in `paconvert/backend/`
- Existing transformers work with both backends through the adapter pattern
- New transformers should use `self.node_to_source(node)` instead of `astor.to_source(node)`

## Architecture

```
paconvert/backend/
├── __init__.py              # Backend module exports
├── base_backend.py          # Abstract base class
├── astor_backend.py         # Astor implementation
├── libcst_backend.py        # Libcst implementation
├── backend_manager.py       # Backend factory and management
└── transformer_adapter.py   # Adapter for existing transformers
```

## Error Handling

- If libcst is not installed and libcst backend is requested, the system will show an error with installation instructions
- If an invalid backend is specified, the system will show available options
- The system gracefully falls back to astor if libcst is not available (with a warning)

## Testing

Run the verification script to ensure backends are working:

```bash
python verify_backend.py
```

## Future Enhancements

- Native libcst transformers for better performance
- Additional backends (e.g., using ast.unparse for Python 3.9+)
- Enhanced comment and docstring preservation
- Better whitespace handling