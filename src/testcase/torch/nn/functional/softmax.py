
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.softmax)
class SoftmaxTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_softmax_correctness(self):
        input_data = torch.randn(10, 10)
        dim = random.randint(0, 9)
        result = torch.nn.functional.softmax(input_data, dim)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_softmax_large_scale(self):
        input_data = torch.randn(1000, 1000)
        dim = random.randint(0, 999)
        result = torch.nn.functional.softmax(input_data, dim)
        return result

