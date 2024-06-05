
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.polar)
class TorchPolarTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_polar_correctness(self):
        abs = torch.rand(random.randint(1, 10))
        angle = torch.rand(random.randint(1, 10))
        result = torch.polar(abs, angle)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_polar_large_scale(self):
        abs = torch.rand(random.randint(1000, 10000))
        angle = torch.rand(random.randint(1000, 10000))
        result = torch.polar(abs, angle)
        return result

