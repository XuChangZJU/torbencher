
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.jit.ONNXTracedModule)
class TorchJitONNXTracedModuleTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_onnx_traced_module_correctness(self):
        onnx_traced_module = torch.jit.ONNXTracedModule(torch.nn.Linear(random.randint(1, 10), random.randint(1, 10)))
        result = onnx_traced_module.type
        return result

    @test_api_version.larger_than("1.1.3")
    def test_onnx_traced_module_large_scale(self):
        onnx_traced_module = torch.jit.ONNXTracedModule(torch.nn.Linear(random.randint(1000, 10000), random.randint(1000, 10000)))
        result = onnx_traced_module.type
        return result

