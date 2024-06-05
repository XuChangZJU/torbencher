
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.Flatten)
class TorchFlattenTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_flatten_correctness(self):
        input_tensor = torch.randn(random.randint(1, 10), random.randint(1, 10), random.randint(1, 10), random.randint(1, 10))
        flatten = torch.nn.Flatten()
        result = flatten(input_tensor)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_flatten_large_scale(self):
        input_tensor = torch.randn(random.randint(1000, 10000), random.randint(100, 1000), random.randint(100, 1000), random.randint(100, 1000))
        flatten = torch.nn.Flatten()
        result = flatten(input_tensor)
        return result

