
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.Unflatten)
class TorchUnflattenTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_unflatten_correctness(self):
        dim = random.randint(1, 10)
        unflatten_shape = (random.randint(1, 10), random.randint(1, 10))
        input_tensor = torch.randn(random.randint(1, 10), random.randint(1, 10) * random.randint(1, 10))
        unflatten = torch.nn.Unflatten(dim=dim, unflatten_shape=unflatten_shape)
        result = unflatten(input_tensor)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_unflatten_large_scale(self):
        dim = random.randint(1, 10)
        unflatten_shape = (random.randint(100, 1000), random.randint(100, 1000))
        input_tensor = torch.randn(random.randint(1000, 10000), random.randint(100, 1000) * random.randint(100, 1000))
        unflatten = torch.nn.Unflatten(dim=dim, unflatten_shape=unflatten_shape)
        result = unflatten(input_tensor)
        return result

