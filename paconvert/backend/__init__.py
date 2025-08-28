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

from .manager import BackendManager
from .base import BaseBackend
from .validation import validate_backend_availability, handle_backend_validation_error, create_backend_manager_safely

__all__ = ["BackendManager", "BaseBackend", "validate_backend_availability", 
           "handle_backend_validation_error", "create_backend_manager_safely"]