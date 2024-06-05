
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.jit.RecursiveScriptModule)
class TorchJitRecursiveScriptModuleTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_recursive_script_module_correctness(self):
        recursive_script_module = torch.jit.RecursiveScriptModule(torch.nn.Linear(random.randint(1, 10), random.randint(1, 10)))
        result = recursive_script_module.type
        return result

    @test_api_version.larger_than("1.1.3")
    def test_recursive_script_module_large_scale(self):
        recursive_script_module = torch.jit.RecursiveScriptModule(torch.nn.Linear(random.randint(1000, 10000), random.randint(1000, 10000)))
        result = recursive_script_module.type
        return result

