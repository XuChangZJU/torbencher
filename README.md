# torbencher

pytorch在各GPU/TPU/NPU上的兼容性和基础性能评估

## 使用方法：

先`git clone`项目到本地

在厂商已经设置好的pytorch环境的计算机上执行如下代码：

```python
from torbencher import torbencher

bencher = torbencher({
    "seed": 1234567890,
    "devices": [
        "cuda"
    ],
    "test_modules": [
        "torch",
        "torch.nn",
        "torch.nn.functional"
        # ...
    ],
    "format": "json"
})
result = bencher.run()
print(result)
```

程序将测试环境所配置的pytorch相应版本的接口，输出类似如下的结果：

```json
{
  "start_time": "2024-05-31 9:11:12",
  "cpu": "Intel Core i9-13900H",
  "memory": "1024GB",
  "test_modules": "all",
  "version": "2.1.0",
  "results": [
    {
      "device": "mlu",
      "cost": "00:04:28.324",
      "packages": 21,
      "apis": 1429,
      "api_passed": 1420,
      "api_failed": 9,
      "api_missed": 1211,
      "detail": {
        "torch": {
          "add": {
            "add_test_1d": [
              0.0021,
              [
                1
              ]
            ],
            "add_test_11d": [
              0.0031,
              [
                11
              ]
            ],
            "add_test_complicated": "failed"
          },
          "addcdiv": "missed"
        }
      }
    }
  ]
}
```

## 配置说明

- `seed`: 随机种子，默认为`time.time_ns()`。
- `devices`: 设备名，由厂商指定，框架将对每个设备依次调用`torch.device(name)`并进行测试，CPU~~的指定可以省略~~**无需指定**。
- `test_modules`：要测试的包名，默认是对应版本的所有包（仅限框架目前所支持的）
- `format`：输出格式目前仅支持`json`

## 结果说明

对于每个包的每个接口，会有若干测试用例，每个测试用例的测试结果会同CPU运行结果比对较以确定其正确性。
如果测试通过，其输出格式为：`[运行时间(ms), 测试规模(scale)]`，如果测试不通过，则输出`failed`，如果该API不存在，则输出`missed`

## 测试scale

为了进行大致的性能评估，我们对测试接口的输入参数进行scale度量，度量方法为：

- 如果参数为tensor，取`tensor.size()`
- 其余待定

## 测试用例编写规范

### 目录组织

- 测试用例统一放置在`src/testcase/包名`目录下，例如`torch`包的用例全部放在`src/testcase/torch`目录下，`torch.nn`包的用例全部放在`src/testcase/torch/nn`目录下
- 每个包的目录下的每个文件代表一个接口的测试用例，其命名按照接口的驼峰命名，例如`torch`包的`is_tensor`
  接口的测试用例编写在`src/testcase/torch/isTensor.py`

### 代码规范（2024年7月27日更新，完善中）

一个接口的测试用例如下（以`torch.add`为例）：

```python
"""add.py"""

import torch
import src.util.test_api_version as test_api_version
from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase


class TorchAddTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_add_4d(self, input=None):
        if input is not None:
            result = torch.add(input[0], input[1], input[2])
            return [result, input]
        a = torch.randn(4)
        b = torch.randn(4)
        result = torch.add(a, b, alpha=10)
        return result            # 仅返回结果，不需要像以前一样其他参数什么的
```
**（以下不再需要，只需要所有`__init.py`文件改为通用脚本即可，具体脚本见`src.README.md`）**  
测试用例编写完成后，在对应模块的`__init__.py`文件中导出该用例（如果不导出该用例，框架将不会发现并测试它），如
```python
from .add import TorchAddTestCase
```

要点：
1. 当前已通过扩展`unittest`的loader和runner的方法实现了对测试用例名称与测试用例计算结果的获取，应将所有的`@test_api(api)`删除，
2. 目前已完成稳定器注入的`random`方法已有`random.randint`, `random.uniform`, `torch.randn`和`torch.normal`，在`SingleTester`未进行相应适配前尽可能使用已适配的方法
3. `torch.nn.modules`里的神经网络内部的权重初始化暂时要求以`torch.randn`的形式自行初始化，稳定器注入遇到困难：
 - 初始化方法（以`torch.nn.Linear`为例）
 - 可使用`torch.randn`
```python
with torch.no_grad():
    linear_layer.weight = torch.nn.Parameter(torch.randn(out_features, in_features) * 0.01);
    linear_layer.bias = torch.nn.Parameter(torch.randn(out_features) * 0.01);
```
或使用`torch.normal`（**建议**，使用`torch.normal`往往**更好更规范**）  
```python
with torch.no_grad():
    linear_layer.weight = torch.nn.Parameter(torch.normal(0, 0.01, size=(out_features, in_features)));
    linear_layer.bias = torch.nn.Parameter(torch.normal(0, 0.01, size=(out_features,)));
```
5. **返回值仅包含测试方法返回值，为None的自行判断正确性，但也可通过`SingleTester`测试其语法正确性**
6. 如果用例有pytorch版本限制，可以通过`@api_test_version.larger_than(ver)/less_than(ver)/between(low, high)/equal(ver)`的装饰器来标定范围
7. 
