
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.constant_pad_nd)
class TorchConstantPadNdTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_constant_pad_nd_correctness(self):
        dim = random.randint(1, 10)
        input = torch.randn(dim)
        pad = [random.randint(0, 10) for _ in range(dim * 2)]
        value = random.uniform(0.1, 10.0)
        result = torch.constant_pad_nd(input, pad, value=value)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_constant_pad_nd_large_scale(self):
        dim = random.randint(100, 1000)
        input = torch.randn(dim)
        pad = [random.randint(0, 10) for _ in range(dim * 2)]
        value = random.uniform(0.1, 10.0)
        result = torch.constant_pad_nd(input, pad, value=value)
        return result

