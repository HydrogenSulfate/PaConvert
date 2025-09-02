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

"""
Adapter to inject backend functionality into existing transformers.
"""

def inject_backend_into_transformer(transformer_class, backend):
    """
    Inject backend functionality into a transformer class.
    This allows existing transformers to use the backend for node-to-source conversion.
    """
    # Store original methods
    original_init = transformer_class.__init__
    
    def new_init(self, *args, **kwargs):
        # Call original init
        original_init(self, *args, **kwargs)
        # Inject backend
        self._backend = backend
    
    # Replace __init__ method
    transformer_class.__init__ = new_init
    
    # Add backend-aware node_to_source method
    def node_to_source(self, node):
        return self._backend.node_to_source(node)
    
    transformer_class.node_to_source = node_to_source
    
    return transformer_class