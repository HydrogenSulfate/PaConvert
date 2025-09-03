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

from typing import Union

try:
    import libcst as cst
    from libcst import matchers as m
except ImportError:
    cst = None
    m = None

from .base_transformer import LibcstBaseTransformer
from paconvert.utils import log_info


class LibcstTensorRequiresGradTransformer(LibcstBaseTransformer):
    """Transformer for handling tensor requires_grad attribute using libcst."""
    
    def leave_Attribute(self, original_node: cst.Attribute, updated_node: cst.Attribute) -> cst.Attribute:
        """Transform requires_grad attribute access."""
        if updated_node.attr.value == "requires_grad":
            # Check if this is a tensor attribute
            # For now, we'll keep it as-is since requires_grad handling
            # is complex and depends on context
            pass
        
        return updated_node