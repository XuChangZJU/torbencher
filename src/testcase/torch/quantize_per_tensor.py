
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.quantize_per_tensor)
class TorchQuantizePerTensorTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_quantize_per_tensor_correctness(self):
        tensor = torch.randn(random.randint(1, 10))
        scale = random.uniform(0.1, 10.0)
        zero_point = random.randint(0, 10)
        result = torch.quantize_per_tensor(tensor, scale, zero_point)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_quantize_per_tensor_large_scale(self):
        tensor = torch.randn(random.randint(1000, 10000))
        scale = random.uniform(0.1, 10.0)
        zero_point = random.randint(0, 1000)
        result = torch.quantize_per_tensor(tensor, scale, zero_point)
        return result

