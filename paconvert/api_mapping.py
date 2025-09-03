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
API mapping from PyTorch to PaddlePaddle
"""

# Basic API mappings for demonstration
API_MAPPING = {
    # Basic tensor operations
    "torch.tensor": {
        "paddle_api": "paddle.to_tensor",
        "kwargs_change": {},
    },
    "torch.zeros": {
        "paddle_api": "paddle.zeros",
        "kwargs_change": {},
    },
    "torch.ones": {
        "paddle_api": "paddle.ones", 
        "kwargs_change": {},
    },
    "torch.randn": {
        "paddle_api": "paddle.randn",
        "kwargs_change": {},
    },
    "torch.rand": {
        "paddle_api": "paddle.rand",
        "kwargs_change": {},
    },
    "torch.randint": {
        "paddle_api": "paddle.randint",
        "kwargs_change": {},
    },
    "torch.add": {
        "paddle_api": "paddle.add",
        "kwargs_change": {},
    },
    "torch.matmul": {
        "paddle_api": "paddle.matmul",
        "kwargs_change": {},
    },
    "torch.cat": {
        "paddle_api": "paddle.concat",
        "kwargs_change": {"dim": "axis"},
    },
    
    # Neural network modules
    "torch.nn.Module": {
        "paddle_api": "paddle.nn.Layer",
    },
    "torch.nn.Linear": {
        "paddle_api": "paddle.nn.Linear",
        "kwargs_change": {},
    },
    "torch.nn.Conv2d": {
        "paddle_api": "paddle.nn.Conv2D",
        "kwargs_change": {},
    },
    "torch.nn.ReLU": {
        "paddle_api": "paddle.nn.ReLU",
        "kwargs_change": {},
    },
    "torch.nn.MaxPool2d": {
        "paddle_api": "paddle.nn.MaxPool2D",
        "kwargs_change": {},
    },
    "torch.nn.Dropout": {
        "paddle_api": "paddle.nn.Dropout",
        "kwargs_change": {},
    },
    
    # Functional operations
    "torch.nn.functional.relu": {
        "paddle_api": "paddle.nn.functional.relu",
        "kwargs_change": {},
    },
    "torch.nn.functional.softmax": {
        "paddle_api": "paddle.nn.functional.softmax",
        "kwargs_change": {},
    },
    
    # Tensor methods
    "torch.Tensor.sum": {
        "paddle_api": "paddle.sum",
    },
    "torch.Tensor.mean": {
        "paddle_api": "paddle.mean", 
    },
    "torch.Tensor.view": {
        "paddle_api": "paddle.reshape",
    },
    "torch.Tensor.size": {
        "paddle_api": "paddle.shape",
    },
    
    # Optimizers
    "torch.optim.SGD": {
        "paddle_api": "paddle.optimizer.SGD",
        "kwargs_change": {"lr": "learning_rate"},
        "paddle_default_kwargs": {"weight_decay": 0.0},
    },
    "torch.optim.Adam": {
        "paddle_api": "paddle.optimizer.Adam", 
        "kwargs_change": {"lr": "learning_rate"},
    },
    
    # Loss functions
    "torch.nn.CrossEntropyLoss": {
        "paddle_api": "paddle.nn.CrossEntropyLoss",
    },
    "torch.nn.MSELoss": {
        "paddle_api": "paddle.nn.MSELoss",
    },
}

# Add more mappings as needed...