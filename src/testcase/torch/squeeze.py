
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.squeeze)
class TorchSqueezeTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_squeeze_correctness(self):
        input = torch.randn(random.randint(1, 10), 1, random.randint(1, 10))
        dim = random.randint(0, 2)
        result = torch.squeeze(input, dim)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_squeeze_large_scale(self):
        input = torch.randn(random.randint(1000, 10000), 1, random.randint(1000, 10000))
        dim = random.randint(0, 2)
        result = torch.squeeze(input, dim)
        return result

