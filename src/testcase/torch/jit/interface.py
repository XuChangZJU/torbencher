
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.jit.interface)
class TorchJitInterfaceTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_interface_correctness(self):
        class TestInterface(torch.jit.interface):
            def __init__(self):
                super().__init__()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                raise NotImplementedError

        result = TestInterface()
        return result

    @test_api_version.larger_than("1.1.3")
    def test_interface_large_scale(self):
        class TestInterface(torch.jit.interface):
            def __init__(self):
                super().__init__()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                raise NotImplementedError

        result = TestInterface()
        return result

