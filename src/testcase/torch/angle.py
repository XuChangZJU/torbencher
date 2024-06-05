
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.angle)
class TorchAngleTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_angle_correctness(self):
        input = torch.randn(random.randint(1, 10), dtype=torch.complex64)
        result = torch.angle(input)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_angle_large_scale(self):
        input = torch.randn(random.randint(1000, 10000), dtype=torch.complex128)
        result = torch.angle(input)
        return result

