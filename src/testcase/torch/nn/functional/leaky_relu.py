
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.leaky_relu)
class LeakyReluTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_leaky_relu_correctness(self):
        input_data = torch.randn(10, 10)
        negative_slope = random.uniform(0.0, 1.0)
        result = torch.nn.functional.leaky_relu(input_data, negative_slope)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_leaky_relu_large_scale(self):
        input_data = torch.randn(1000, 1000)
        negative_slope = random.uniform(0.0, 1.0)
        result = torch.nn.functional.leaky_relu(input_data, negative_slope)
        return result

