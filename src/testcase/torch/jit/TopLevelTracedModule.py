
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.jit.TopLevelTracedModule)
class TorchJitTopLevelTracedModuleTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_top_level_traced_module_correctness(self):
        top_level_traced_module = torch.jit.TopLevelTracedModule(torch.nn.Linear(random.randint(1, 10), random.randint(1, 10)))
        result = top_level_traced_module.type
        return result

    @test_api_version.larger_than("1.1.3")
    def test_top_level_traced_module_large_scale(self):
        top_level_traced_module = torch.jit.TopLevelTracedModule(torch.nn.Linear(random.randint(1000, 10000), random.randint(1000, 10000)))
        result = top_level_traced_module.type
        return result

