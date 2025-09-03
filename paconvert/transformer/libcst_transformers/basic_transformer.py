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

from typing import Union, Optional, Dict, Any

try:
    import libcst as cst
    from libcst import matchers as m
except ImportError:
    cst = None
    m = None

from .base_transformer import LibcstBaseTransformer
from paconvert.global_var import GlobalManager
from paconvert.utils import log_info, log_debug


class LibcstBasicTransformer(LibcstBaseTransformer):
    """Basic transformer for converting torch API calls to paddle using libcst."""
    
    def leave_ClassDef(self, original_node: cst.ClassDef, updated_node: cst.ClassDef) -> cst.ClassDef:
        """Transform class definitions, particularly base classes."""
        if updated_node.bases:
            new_bases = []
            changed = False
            
            for base in updated_node.bases:
                if isinstance(base, cst.Arg) and base.value:
                    # Handle base class references
                    new_base_value = self._transform_base_class(base.value)
                    if new_base_value != base.value:
                        changed = True
                    new_bases.append(base.with_changes(value=new_base_value))
                else:
                    new_bases.append(base)
            
            if changed:
                return updated_node.with_changes(bases=new_bases)
        
        return updated_node
    
    def leave_Attribute(self, original_node: cst.Attribute, updated_node: cst.Attribute) -> Union[cst.Attribute, cst.Name]:
        """Transform attribute access that might be torch APIs."""
        # This handles cases like accessing torch.nn.Module as a type annotation
        # or other non-call contexts
        full_name = self.get_full_attr_name(updated_node)
        
        if not self.is_torch_api(updated_node):
            return updated_node
        
        # Map the local name back to the full torch API name
        torch_api = self._resolve_torch_api_name(full_name)
        
        # Check if this API has a mapping
        if torch_api not in GlobalManager.API_MAPPING:
            return updated_node
        
        # Get the mapping configuration
        mapping_config = GlobalManager.API_MAPPING[torch_api]
        paddle_api = mapping_config.get("paddle_api")
        
        if not paddle_api:
            return updated_node
        
        # Create new paddle reference
        new_attr = self._create_paddle_func_ref(paddle_api)
        
        # Log the conversion
        self.torch_api_count += 1
        self.success_api_count += 1
        if self.logger:
            log_info(self.logger, f"[{self.file_name}] [Success] Convert attribute {torch_api} to {paddle_api}")
        
        return new_attr
    
    def leave_Call(self, original_node: cst.Call, updated_node: cst.Call) -> Union[cst.Call, cst.BaseExpression]:
        """Transform function calls from torch to paddle."""
        # Get the full API name for debugging
        full_name = self.get_full_attr_name(updated_node.func)
        
        if not self.is_torch_api(updated_node.func):
            return updated_node
        
        # Map the local name back to the full torch API name
        torch_api = self._resolve_torch_api_name(full_name)
        
        # Check if this API has a mapping
        if torch_api not in GlobalManager.API_MAPPING:
            # Log unsupported API
            self.torch_api_count += 1
            if self.unsupport_api_map is not None:
                self.unsupport_api_map[torch_api] += 1
            if self.logger:
                log_debug(self.logger, f"[{self.file_name}] [Not Support] {torch_api} is not supported")
            return updated_node
        
        # Get the mapping configuration
        mapping_config = GlobalManager.API_MAPPING[torch_api]
        paddle_api = mapping_config.get("paddle_api")
        
        if not paddle_api:
            self.torch_api_count += 1
            if self.unsupport_api_map is not None:
                self.unsupport_api_map[torch_api] += 1
            return updated_node
        
        # Transform the API call
        try:
            new_call = self._transform_api_call(updated_node, torch_api, mapping_config)
            if new_call != updated_node:
                self.torch_api_count += 1
                self.success_api_count += 1
                if self.logger:
                    log_info(self.logger, f"[{self.file_name}] [Success] Convert {torch_api} to Paddle")
                return new_call
        except Exception as e:
            if self.logger:
                log_debug(self.logger, f"[{self.file_name}] [Error] Failed to convert {torch_api}: {e}")
        
        self.torch_api_count += 1
        if self.unsupport_api_map is not None:
            self.unsupport_api_map[torch_api] += 1
        return updated_node
    
    def _transform_api_call(self, call_node: cst.Call, torch_api: str, mapping_config: Dict[str, Any]) -> cst.Call:
        """Transform a specific API call based on mapping configuration."""
        paddle_api = mapping_config["paddle_api"]
        
        # Step 1: Extract 'out' parameter if present
        out_arg = None
        filtered_args = []
        
        for arg in call_node.args:
            if isinstance(arg, cst.Arg) and arg.keyword and arg.keyword.value == "out":
                out_arg = arg
                # Don't add 'out' to filtered_args
            else:
                filtered_args.append(arg)
        
        # Step 2: Transform remaining arguments
        transformed_args = []
        kwargs_change = mapping_config.get("kwargs_change", {})
        
        for arg in filtered_args:
            if isinstance(arg, cst.Arg) and arg.keyword:
                keyword_name = arg.keyword.value
                if keyword_name in kwargs_change:
                    new_keyword = kwargs_change[keyword_name]
                    if new_keyword:  # Only rename if not empty
                        new_arg = arg.with_changes(keyword=cst.Name(new_keyword))
                        transformed_args.append(new_arg)
                    # If empty, skip this argument
                else:
                    # Keep other keyword arguments as-is
                    transformed_args.append(arg)
            else:
                # Keep positional arguments as-is
                transformed_args.append(arg)
        
        # Step 3: Create new function reference
        new_func = self._create_paddle_func_ref(paddle_api)
        
        # Step 4: Create the basic paddle call
        new_call = call_node.with_changes(func=new_func, args=transformed_args)
        
        # Step 5: If there's an 'out' parameter, wrap with paddle.assign
        if out_arg:
            assign_func = cst.Attribute(value=cst.Name("paddle"), attr=cst.Name("assign"))
            assign_args = [
                cst.Arg(value=new_call),  # The result of the paddle function
                cst.Arg(keyword=cst.Name("output"), value=out_arg.value)  # The output tensor
            ]
            return cst.Call(func=assign_func, args=assign_args)
        
        return new_call
    
    def _create_paddle_func_ref(self, paddle_api: str) -> cst.BaseExpression:
        """Create a paddle function reference from API string."""
        parts = paddle_api.split('.')
        
        if len(parts) == 1:
            return cst.Name(parts[0])
        
        # Build nested attribute access
        result = cst.Name(parts[0])
        for part in parts[1:]:
            result = cst.Attribute(value=result, attr=cst.Name(part))
        
        return result
    
    def _transform_arguments(self, args: list, mapping_config: Dict[str, Any]) -> list:
        """Transform function arguments based on mapping configuration."""
        kwargs_change = mapping_config.get("kwargs_change", {})
        
        if self.logger:
            log_debug(self.logger, f"kwargs_change: {kwargs_change}")
        
        # Parameters that should be removed (not passed to paddle API)
        remove_params = {
            "layout", "device", "memory_format", "inplace", "generator", 
            "non_blocking", "async", "dtype", "pin_memory", "requires_grad"
        }
        
        new_args = []
        for arg in args:
            if isinstance(arg, cst.Arg) and arg.keyword:
                # Handle keyword arguments
                keyword_name = arg.keyword.value
                
                if self.logger:
                    log_debug(self.logger, f"Processing keyword arg: {keyword_name}")
                
                # Skip parameters that should be removed
                if keyword_name in remove_params:
                    if self.logger:
                        log_debug(self.logger, f"Removing parameter: {keyword_name}")
                    continue
                
                if keyword_name in kwargs_change:
                    # Rename the keyword
                    new_keyword = kwargs_change[keyword_name]
                    if self.logger:
                        log_debug(self.logger, f"Renaming {keyword_name} to {new_keyword}")
                    if new_keyword:  # Skip if mapped to empty string (removal)
                        new_arg = arg.with_changes(keyword=cst.Name(new_keyword))
                        new_args.append(new_arg)
                    else:
                        if self.logger:
                            log_debug(self.logger, f"Removing parameter {keyword_name} (mapped to empty)")
                    # If mapped to empty string, skip (don't add to new_args)
                else:
                    # Keep other keyword arguments as-is
                    if self.logger:
                        log_debug(self.logger, f"Keeping parameter as-is: {keyword_name}")
                    new_args.append(arg)
            else:
                # Handle positional arguments - always keep them
                if self.logger:
                    log_debug(self.logger, f"Keeping positional arg: {arg}")
                new_args.append(arg)
        
        # Add default paddle arguments if specified
        paddle_defaults = mapping_config.get("paddle_default_kwargs", {})
        for key, value in paddle_defaults.items():
            # Check if this argument is already provided
            has_arg = any(
                isinstance(arg, cst.Arg) and arg.keyword and arg.keyword.value == key
                for arg in new_args
            )
            if not has_arg:
                # Add default argument
                default_value = self._create_value_node(value)
                new_args.append(cst.Arg(keyword=cst.Name(key), value=default_value))
        
        return new_args
    
    def _transform_base_class(self, base_value: cst.BaseExpression) -> cst.BaseExpression:
        """Transform base class references from torch to paddle."""
        full_name = self.get_full_attr_name(base_value)
        
        if not self.is_torch_api(base_value):
            return base_value
        
        # Map the local name back to the full torch API name
        torch_api = self._resolve_torch_api_name(full_name)
        
        # Check if this API has a mapping
        if torch_api not in GlobalManager.API_MAPPING:
            return base_value
        
        # Get the mapping configuration
        mapping_config = GlobalManager.API_MAPPING[torch_api]
        paddle_api = mapping_config.get("paddle_api")
        
        if not paddle_api:
            return base_value
        
        # Create new paddle reference
        new_base = self._create_paddle_func_ref(paddle_api)
        
        # Log the conversion
        self.torch_api_count += 1
        self.success_api_count += 1
        if self.logger:
            log_info(self.logger, f"[{self.file_name}] [Success] Convert base class {torch_api} to {paddle_api}")
        
        return new_base
    
    def _resolve_torch_api_name(self, local_name: str) -> str:
        """Resolve a local API name to the full torch API name."""
        if self.file not in self.imports_map:
            return local_name
        
        # Debug print
        if self.logger:
            log_debug(self.logger, f"Resolving {local_name}, imports_map: {self.imports_map[self.file]}")
        
        # Check if we have a direct mapping for this local name
        if local_name in self.imports_map[self.file]:
            full_torch_name = self.imports_map[self.file][local_name]
            resolved = local_name.replace(local_name.split('.')[0], full_torch_name, 1)
            if self.logger:
                log_debug(self.logger, f"Direct mapping: {local_name} -> {resolved}")
            return resolved
        
        # Check if the first part is a torch package
        first_part = local_name.split('.')[0]
        if first_part in self.imports_map[self.file]:
            full_torch_name = self.imports_map[self.file][first_part]
            resolved = local_name.replace(first_part, full_torch_name, 1)
            if self.logger:
                log_debug(self.logger, f"First part mapping: {local_name} -> {resolved}")
            return resolved
        
        if self.logger:
            log_debug(self.logger, f"No mapping found for: {local_name}")
        return local_name
    
    def _create_value_node(self, value: Any) -> cst.BaseExpression:
        """Create a CST node for a given value."""
        if isinstance(value, str):
            return cst.SimpleString(f'"{value}"')
        elif isinstance(value, (int, float)):
            return cst.Integer(str(value)) if isinstance(value, int) else cst.Float(str(value))
        elif isinstance(value, bool):
            return cst.Name("True" if value else "False")
        elif value is None:
            return cst.Name("None")
        else:
            # Fallback: convert to string
            return cst.SimpleString(f'"{str(value)}"')