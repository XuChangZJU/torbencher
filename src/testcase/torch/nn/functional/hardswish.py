
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.hardswish)
class HardswishTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_hardswish_correctness(self):
        input_data = torch.randn(10, 10)
        result = torch.nn.functional.hardswish(input_data)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_hardswish_large_scale(self):
        input_data = torch.randn(1000, 1000)
        result = torch.nn.functional.hardswish(input_data)
        return result

