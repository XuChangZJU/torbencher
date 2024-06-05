
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.LeakyReLU)
class TorchLeakyReLUTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_leakyrelu_correctness(self):
        negative_slope = random.uniform(0.1, 10.0)
        input_tensor = torch.randn(random.randint(1, 10), random.randint(1, 10))
        leaky_relu = torch.nn.LeakyReLU(negative_slope=negative_slope)
        result = leaky_relu(input_tensor)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_leakyrelu_large_scale(self):
        negative_slope = random.uniform(0.1, 10.0)
        input_tensor = torch.randn(random.randint(1000, 10000), random.randint(100, 1000))
        leaky_relu = torch.nn.LeakyReLU(negative_slope=negative_slope)
        result = leaky_relu(input_tensor)
        return result

