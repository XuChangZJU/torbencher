
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.instance_norm)
class InstanceNormTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_instance_norm_correctness(self):
        num_features = random.randint(1, 10)
        input_data = torch.randn(10, num_features, 10, 10)
        weight = torch.randn(num_features)
        bias = torch.randn(num_features)
        eps = random.uniform(0.0, 1.0)
        result = torch.nn.functional.instance_norm(input_data, weight, bias, eps)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_instance_norm_large_scale(self):
        num_features = random.randint(100, 1000)
        input_data = torch.randn(100, num_features, 100, 100)
        weight = torch.randn(num_features)
        bias = torch.randn(num_features)
        eps = random.uniform(0.0, 1.0)
        result = torch.nn.functional.instance_norm(input_data, weight, bias, eps)
        return result

