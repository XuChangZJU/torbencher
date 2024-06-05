
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.PReLU)
class TorchPReLUTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_prelu_correctness(self):
        input_tensor = torch.randn(random.randint(1, 10), random.randint(1, 10))
        prelu = torch.nn.PReLU()
        result = prelu(input_tensor)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_prelu_large_scale(self):
        input_tensor = torch.randn(random.randint(1000, 10000), random.randint(100, 1000))
        prelu = torch.nn.PReLU()
        result = prelu(input_tensor)
        return result

