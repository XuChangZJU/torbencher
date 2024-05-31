# torbencher

pytorch在各GPU/TPU/NPU上的兼容性和基础性能评估

## 使用方法：

先`git clone`项目到本地

在厂商已经设置好的pytorch环境的计算机上执行如下代码：

```python
from path / to / torbencher
import src

src.config({
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
result = src.run()
print(result)
```

程序将测试环境所配置的pytorch相应版本的接口，输出类似如下的结果：
```json
{
    "date": "2024-05-31 9:11",
    "cost": "00:04:28.324",
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
                        "add_test_1d": [0.0021, [1]],
                        "add_test_11d": [0.0031, [11]],
                        "add_test_complicated": "failed"
                    },
                    "addcdiv": "missed",
                }
            }
        }
    ]
}
```
## 配置说明

- `seed`: 随机种子，默认为`time.time_ns()`。
- `devices`: 设备名，由厂商指定，框架将对每个设备依次调用`torch.device(name)`并进行测试。
- `test_modules`：要测试的包名，默认是对应版本的所有包（仅限框架目前所支持的）
- `format`：输出格式 `json/html`

## 结果说明
对于每个包的每个接口，会有若干测试用例，每个测试用例的测试结果会同CPU运行结果比对较以确定其正确性。如果通过，其输出格式为：`[运行时间(ms), 测试规模(scale)]`，如果不通过，则输出`failed`

## 测试scale
为了进行大致的性能评估，我们对测试接口的输入参数进行scale度量，度量方法为：

- 如果参数为tensor，取`tensor.size()`
- 待定


## 测试用例编写规范

### 目录组织 
- 测试用例统一放置在`src/testcase/包名`目录下，例如`torch`包的用例全部放置在`src/testcase/torch`目录下
- 每个包的目录下的每个文件代表一个接口的测试用例，其命名按照接口的驼峰命名，例如`torch`包的`is_tensor`接口的测试用例编写在`src/testcase/torch/isTensor.py`

### 代码规范
一个接口的测试用例如下（以`torch.add`为例）：

```python
import torch
from TorBencherBase import TorBencherBase
from decorator import api_test

@api_test(torch.add)
class TorchAddTest(TorBencherBase):
    @api_test.version.largerThan("1.1.3")
    def test_add_4d(input = None):
        if input is not None:
            result = torch.add(input[0], input[1], input[2])
            return [result, input]
        a = torch.randn(4)
        b = torch.randn(4)
        result = torch.add(a, b, 10)
        return [result, [a, b, 10]]
```

要点：
- 每个用例的返回值格式为：
```
[结果，调用参数]
```
其中，结果目前只允许返回以下类型：`Tensor/Boolean/Number/String/List/Set`

调用参数是调用要测试的接口的参数输入，以数组方式返回。如果测试本接口不需要任何参数输入，可返回`None`

- 每个用例的输入参数可接受返回的参数格式，如输入参数为None，则可以自由随机生成测试参数（但需要返回）
- 如果用例有pytorch版本限制，可以通过`@api_test_version.(largerThan/lessThan/between/equal)`的装饰器来标定范围。