
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.LayerNorm)
class TorchLayerNormTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_layernorm_correctness(self):
        normalized_shape = (random.randint(1, 10),)
        input_tensor = torch.randn(random.randint(1, 10), random.randint(1, 10), random.randint(1, 10))
        layer_norm = torch.nn.LayerNorm(normalized_shape)
        result = layer_norm(input_tensor)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_layernorm_large_scale(self):
        normalized_shape = (random.randint(100, 1000),)
        input_tensor = torch.randn(random.randint(1000, 10000), random.randint(100, 1000), random.randint(100, 1000))
        layer_norm = torch.nn.LayerNorm(normalized_shape)
        result = layer_norm(input_tensor)
        return result

