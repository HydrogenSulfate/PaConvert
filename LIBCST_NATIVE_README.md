# PaConvert 原生 LibCST 后端实现

## 概述

本项目实现了基于原生 LibCST 的 PyTorch 到 PaddlePaddle 代码转换后端。与传统的 AST/Astor 方案不同，这个实现完全基于 LibCST 的 CST（Concrete Syntax Tree）操作，能够完美保留代码的注释、格式和风格。

## 🎯 核心特性

### ✅ 完全原生的 LibCST 实现
- **零依赖 AST/Astor**：完全摆脱对 `ast` 和 `astor` 库的依赖
- **原生 CST 操作**：直接使用 `libcst.CSTTransformer` 和相关接口
- **类型安全**：利用 LibCST 的完整类型系统

### ✅ 注释和格式保留
```python
# 输入代码
import torch  # PyTorch library
x = torch.tensor([1, 2, 3])  # Create tensor

# 输出代码（注释完全保留）
import paddle  # PyTorch library  
x = paddle.to_tensor([1, 2, 3])  # Create tensor
```

### ✅ 模块化架构
```
paconvert/
├── backend/
│   ├── libcst_backend.py        # 原生 LibCST 后端
│   └── ...
├── transformer/libcst_transformers/  # 专门的 LibCST 转换器
│   ├── base_transformer.py     # 继承 cst.CSTTransformer
│   ├── basic_transformer.py    # 原生 CST API 转换
│   ├── import_transformer.py   # 原生 CST 导入处理
│   └── ...
└── api_mapping.py              # 统一的 API 映射配置
```

## 🚀 快速开始

### 安装依赖
```bash
pip install libcst
```

### 使用方法
```bash
# 使用原生 libcst 后端
paconvert -i torch_code.py -o paddle_code.py --backend libcst

# 仍然支持传统 astor 后端
paconvert -i torch_code.py -o paddle_code.py --backend astor
```

### Python API
```python
from paconvert.converter import Converter

converter = Converter(backend="libcst")
converter.run("input.py", "output.py")
```

## 🏗️ 架构设计

### 1. 后端抽象层
```python
class BaseBackend(ABC):
    @abstractmethod
    def parse_code(self, code: str) -> Any:
        """解析源代码为树表示"""
    
    @abstractmethod 
    def generate_code(self, tree: Any) -> str:
        """从树表示生成源代码"""
    
    @abstractmethod
    def get_backend_type(self) -> str:
        """返回后端类型标识符"""
```

### 2. LibCST 原生后端
```python
class LibcstBackend(BaseBackend):
    def parse_code(self, code: str) -> cst.Module:
        return cst.parse_module(code)
    
    def generate_code(self, tree: cst.Module) -> str:
        return tree.code
    
    def get_backend_type(self) -> str:
        return "cst"
```

### 3. 原生转换器系统
```python
class LibcstBasicTransformer(cst.CSTTransformer):
    def leave_Call(self, original_node: cst.Call, updated_node: cst.Call) -> cst.Call:
        """转换函数调用从 torch 到 paddle"""
        if self.is_torch_api(updated_node.func):
            return self._transform_api_call(updated_node, ...)
        return updated_node
```

## 📋 API 映射系统

### 映射配置格式
```python
API_MAPPING = {
    "torch.tensor": {
        "paddle_api": "paddle.to_tensor",
        "kwargs_change": {"data": "x"},
    },
    "torch.nn.Linear": {
        "paddle_api": "paddle.nn.Linear",
        "kwargs_change": {"in_features": "in_features"},
    },
}
```

### 支持的转换特性
- ✅ 关键字参数重命名
- ✅ 默认值插入
- ✅ 参数类型转换
- ✅ 参数顺序调整

## 🧪 测试和验证

### 运行验证脚本
```bash
python verify_libcst_implementation.py
```

### 运行演示
```bash
python demo_libcst_conversion.py
```

### 测试文件
- `verify_libcst_implementation.py` - 基础功能验证
- `demo_libcst_conversion.py` - 完整转换演示
- `example_with_comments.py` - 注释保留示例

## 📊 性能对比

| 特性 | AST/Astor | 原生 LibCST |
|------|-----------|-------------|
| 注释保留 | ❌ | ✅ |
| 格式保留 | ❌ | ✅ |
| 类型安全 | 部分 | ✅ |
| 性能 | 快 | 快 |
| 可维护性 | 中等 | 高 |
| 扩展性 | 中等 | 高 |

## 🔧 扩展和定制

### 添加新的 API 映射
1. 在 `api_mapping.py` 中添加映射配置
2. 实现特殊的转换逻辑（如需要）
3. 添加测试用例

### 创建自定义转换器
```python
class CustomLibcstTransformer(LibcstBaseTransformer):
    def leave_FunctionDef(self, original_node, updated_node):
        # 自定义函数定义转换逻辑
        return updated_node
```

## 📚 文档

- `LIBCST_NATIVE_IMPLEMENTATION.md` - 详细技术实现文档
- `BACKEND_MIGRATION.md` - 迁移指南
- `REFACTORING_SUMMARY.md` - 重构总结

## 🤝 贡献

欢迎贡献代码！请确保：
1. 遵循现有的代码风格
2. 添加适当的测试
3. 更新相关文档

## 📄 许可证

本项目采用 Apache License 2.0 许可证。

## 🎉 总结

这个原生 LibCST 实现真正实现了：

1. **完全摆脱 AST/Astor 依赖**：LibCST 后端不再使用任何 `ast` 或 `astor` 调用
2. **真正的注释保留**：不丢失任何注释信息
3. **格式保持**：保持原始代码的格式风格
4. **类型安全**：利用 LibCST 的类型系统
5. **高可维护性**：清晰的架构和扩展点
6. **完全兼容**：与现有 astor 后端并存

现在您可以享受真正专业级的 PyTorch 到 PaddlePaddle 代码转换体验！