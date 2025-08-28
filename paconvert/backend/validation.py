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

import sys
import logging
from typing import Tuple, Optional

from .manager import BackendManager, BACKEND_ERRORS


def validate_backend_availability(backend_type: str, logger: Optional[logging.Logger] = None) -> Tuple[bool, str]:
    """Validate if a backend is available for use.
    
    Args:
        backend_type: Backend type to validate
        logger: Optional logger for warnings
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Check if backend type is valid
    if not BackendManager.validate_backend_type(backend_type):
        valid_backends = ", ".join(BackendManager.get_valid_backends())
        error_msg = BACKEND_ERRORS["invalid_backend"].format(backend=backend_type)
        error_msg += f". Valid options: {valid_backends}"
        return False, error_msg
    
    # Check libcst availability if requested
    if backend_type == "libcst":
        try:
            import libcst
            return True, ""
        except ImportError:
            error_msg = BACKEND_ERRORS["libcst_not_installed"]
            return False, error_msg
    
    # astor should always be available (it's a core dependency)
    return True, ""


def handle_backend_validation_error(backend_type: str, error_message: str, 
                                  logger: Optional[logging.Logger] = None) -> None:
    """Handle backend validation errors with appropriate logging and exit.
    
    Args:
        backend_type: The backend type that failed validation
        error_message: The error message to display
        logger: Optional logger instance
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.error(f"Backend validation failed for '{backend_type}': {error_message}")
    print(f"Error: {error_message}", file=sys.stderr)
    
    # Provide helpful suggestions
    if backend_type == "libcst":
        print("\nTo use libcst backend, install it with:", file=sys.stderr)
        print("  pip install libcst", file=sys.stderr)
        print("\nOr use the default astor backend:", file=sys.stderr)
        print("  paconvert --backend astor <other_args>", file=sys.stderr)
    
    sys.exit(1)


def create_backend_manager_safely(backend_type: str, logger: Optional[logging.Logger] = None) -> BackendManager:
    """Create a BackendManager with proper error handling.
    
    Args:
        backend_type: Backend type to create
        logger: Optional logger instance
        
    Returns:
        BackendManager instance
        
    Raises:
        SystemExit: If backend creation fails
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Validate backend availability first
    is_valid, error_msg = validate_backend_availability(backend_type, logger)
    if not is_valid:
        handle_backend_validation_error(backend_type, error_msg, logger)
    
    # Try to create backend manager
    try:
        manager = BackendManager(backend_type, logger)
        
        # Log if fallback was used
        if manager.is_fallback_used():
            logger.warning(f"Requested backend '{manager.get_requested_backend()}' "
                         f"not available, using '{manager.get_backend_name()}' instead")
        
        return manager
        
    except Exception as e:
        error_msg = f"Failed to create backend manager: {str(e)}"
        handle_backend_validation_error(backend_type, error_msg, logger)