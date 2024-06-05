
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.jit.optimized_execution)
class TorchJitOptimizedExecutionTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_optimized_execution_correctness(self):
        result = torch.jit.optimized_execution(random.random() > 0.5)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_optimized_execution_large_scale(self):
        result = torch.jit.optimized_execution(random.random() > 0.5)
        return result

