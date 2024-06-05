
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.jit.script_method)
class TorchJitScriptMethodTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_script_method_correctness(self):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            @torch.jit.script_method
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x + 1

        module = TestModule()
        result = module(torch.randn(random.randint(1, 