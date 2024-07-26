# TorchWrapper

一个对pytorch包进行封装，对调用进行记录的工具

## 使用

先`git clone`项目到本地

将需要运行的程序改写为如下代码：

```python
from torbencher import TorchWrapper

wrapper = Wrapper({
    "out_dir": "path/to/output",
    "format": "csv",
    "file_max_size": "512MB",
    "file_name_spec": "timestamp"
})

def my_code(*args, **kwargs):
    """
    此处为需要运行的程序
    """

wrapper.start(my_code)
```

然后执行程序，对pytorch的调用序列就会被顺序输出到`out_dir`目录下的文件中

## 配置参数

- `out_dir`: 输出目录（必需）
- `format`: 输出格式目前支持`csv`,`json`和`html`，默认为`csv`
- `file_max_size`：输出文件的最大值，超过会自动输出到下一个文件中
- `file_name_spec`：输出文件名规范，`timestamp`时间戳，`datetime`以时间值（字符串），`serial`以6位正整数（如果文件夹下原来就有输出文件，会接着最大文件名命名）

## 输出格式：

csv文件的每行代表一次调用，格式为：

```
接口名,调用次数,调用时间戳,耗费时间(ms),scale字符串(json化)
```

其中scale代表本次调用的参数的scale被JSON.stringify后的字符串，scale的标定规则与bencher保持一致






# bencherDebugger（Torbencher“青春版”，内部使用）

一个供算子测试用例开发过程进行全面Debug的内部工具

## 使用
### 倒入bencherDebugger工具

```python
from torbencher import benchDebugger
```
###定义一个需要进行测试的模块列表
```python
modules = [
"torch.nn.functional", "torch.optim", "torch.special", "torch.random", "torch.utils.cpp_extension", "torch.utils.data",
# "torch.xpu",
# "torch.mps",
"torch.jit", "torch.utils", "torch.distributions", "torch.autograd", "torch", "torch.onnx",
# "torch.cuda",
"torch.linalg", "torch.amp", "torch.nn", "torch.utils.mobile_optimizer", "torch.distributed",
"torch.utils.checkpoint", "torch.Tensor","torch.export", "torch.profiler", "torch.backends", "torch.fx", "torch.cpu",
# "torch.hub",
"torch.testing", "torch.masked","torch.utils.tensorboard","torch.nn.init","torch.fft","torch.autograd"]
```
### 配置bencherDebugger
参数会在后续进行介绍
```python
config = {
        "seed": 1234567890,
        "devices": ["cpu"],
        "test_modules": modules,
        "format": "json",
        "num_epoches": 5,
        "including_success": False
    }
```
### 实例化bencherDebugger工具
```python
debugger = bencherDebugger(config)
```
### 运行bencherDebugger工具
```python
debugger.run()
```

## 配置参数
1. `seed`：对于一些基于随机数测试的用例进行种子固定
2. `devices`：指定需要进行测试的设备（由于TorbencherTestCaseBase暂未适配device指定，故未实装）
3. `test_modules`：指定需要进行测试的torch模块
4. `format`：指定测试结果的输出格式（暂时只支持csv，未实装其它格式）
5. `num_epoches`：指定所有算子测试的轮数（仅全部通过的算字标注为Success，且有必要避免`torch.hub`等需要进行人工选择的测试用例，否则需要每一轮都进行输入）
6. `including_success`：指定结果是否包含通过测试的算子用例

## 输出内容
| 字段名称          | 可能的取值示例                                              |
|-------------------|-----------------------------------------------------------|
| errors            | assert失败的个数                                           |
| error_details     | 报错的具体信息                                             |
| failures          | 非报错未通过点的个数                                        |
| failure_details   |  非报错未通过点的详情                                       |
| status            | Success, Failed, UnitTestError, ModuleImportError.        |
| testcase          | 导入失败的模块名, TorbencherTestCaseBase的子类等            |

# 一个通用`__init__.py`脚本：
该脚本可以放在本工具的任意`__init__.py`中使用，同时增加import时的debug信息以辅助开发过程
```python
import os
import importlib
import logging

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase

# 设置日志配置
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

current_directory: str = os.path.dirname(os.path.abspath(__file__))
script_files: list = [f for f in os.listdir(current_directory) if f.endswith('.py') and f != '__init__.py']

for script_file in script_files:
    module_name: str = script_file[:-3]  # Remove the .py extension
    try:
        module = importlib.import_module(f'.{module_name}', package=__package__)
        # logger.debug(f"Successfully imported module {module_name}")
    except Exception as e:
        logger.debug(f"Failed to import module {module_name}: {e}")
        continue

    for attribute_name in dir(module):
        attribute = getattr(module, attribute_name)
        if isinstance(attribute, type) and issubclass(attribute, TorBencherTestCaseBase):
            globals()[attribute_name] = attribute
```
