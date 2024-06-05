
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.layer_norm)
class LayerNormTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_layer_norm_correctness(self):
        input_data = torch.randn(10, 10, 10)
        normalized_shape = (10, 10)
        weight = torch.randn(10 * 10)
        bias = torch.randn(10 * 10)
        eps = random.uniform(0.0, 1.0)
        result = torch.nn.functional.layer_norm(input_data, normalized_shape, weight, bias, eps)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_layer_norm_large_scale(self):
        input_data = torch.randn(100, 100, 100)
        normalized_shape = (100, 100)
        weight = torch.randn(100 * 100)
        bias = torch.randn(100 * 100)
        eps = random.uniform(0.0, 1.0)
        result = torch.nn.functional.layer_norm(input_data, normalized_shape, weight, bias, eps)
        return result

