
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.is_floating_point)
class TorchIsFloatingPointTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_is_floating_point_correctness(self):
        tensor = torch.randn(random.randint(1, 10), dtype=torch.float32)
        result = torch.is_floating_point(tensor)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_is_floating_point_large_scale(self):
        tensor = torch.randn(random.randint(1000, 10000), dtype=torch.float64)
        result = torch.is_floating_point(tensor)
        return result

