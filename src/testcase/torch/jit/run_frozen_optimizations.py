
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.jit.run_frozen_optimizations)
class TorchJitRunFrozenOptimizationsTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_run_frozen_optimizations_correctness(self):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(random.randint(1, 10), random.randint(1, 10))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.linear(x)

        module = TestModule()
        result = torch.jit.run_frozen_optimizations(module)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_run_frozen_optimizations_large_scale(self):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(random.randint(1000, 10000), random.randint(1000, 10000))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.linear(x)

        module = TestModule()
        result = torch.jit.run_frozen_optimizations(module)
        return result

