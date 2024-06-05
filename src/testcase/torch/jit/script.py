
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.jit.script)
class TorchJitScriptTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_script_correctness(self):
        @torch.jit.script
        def test_func(x: torch.Tensor) -> torch.Tensor:
            return x + 1
        result = test_func(torch.randn(random.randint(1, 10)))
        return result

    @test_api_version.larger_than("1.1.3")
    def test_script_large_scale(self):
        @torch.jit.script
        def test_func(x: torch.Tensor) -> torch.Tensor:
            return x + 1
        result = test_func(torch.randn(random.randint(1000, 10000)))
        return result

