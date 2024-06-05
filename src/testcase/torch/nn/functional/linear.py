
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.linear)
class LinearTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_linear_correctness(self):
        in_features = random.randint(1, 10)
        out_features = random.randint(1, 10)
        input_data = torch.randn(10, in_features)
        weight = torch.randn(out_features, in_features)
        bias = torch.randn(out_features)
        result = torch.nn.functional.linear(input_data, weight, bias)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_linear_large_scale(self):
        in_features = random.randint(100, 1000)
        out_features = random.randint(100, 1000)
        input_data = torch.randn(100, in_features)
        weight = torch.randn(out_features, in_features)
        bias = torch.randn(out_features)
        result = torch.nn.functional.linear(input_data, weight, bias)
        return result

