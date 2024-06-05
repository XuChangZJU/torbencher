
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.Softmax)
class TorchSoftmaxTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_softmax_correctness(self):
        input_tensor = torch.randn(random.randint(1, 10), random.randint(1, 10))
        softmax = torch.nn.Softmax(dim=1)
        result = softmax(input_tensor)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_softmax_large_scale(self):
        input_tensor = torch.randn(random.randint(1000, 10000), random.randint(100, 1000))
        softmax = torch.nn.Softmax(dim=1)
        result = softmax(input_tensor)
        return result

