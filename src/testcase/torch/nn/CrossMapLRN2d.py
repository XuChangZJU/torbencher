
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.CrossMapLRN2d)
class TorchCrossMapLRN2dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_crossmaplrn2d_correctness(self):
        size = random.randint(1, 10)
        alpha = random.uniform(0.1, 10.0)
        beta = random.uniform(0.1, 10.0)
        k = random.uniform(0.1, 10.0)
        input_tensor = torch.randn(random.randint(1, 10), random.randint(1, 10), random.randint(1, 10), random.randint(1, 10))
        cross_map_lrn = torch.nn.CrossMapLRN2d(size=size, alpha=alpha, beta=beta, k=k)
        result = cross_map_lrn(input_tensor)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_crossmaplrn2d_large_scale(self):
        size = random.randint(100, 1000)
        alpha = random.uniform(0.1, 10.0)
        beta = random.uniform(0.1, 10.0)
        k = random.uniform(0.1, 10.0)
        input_tensor = torch.randn(random.randint(1000, 10000), random.randint(100, 1000), random.randint(100, 1000), random.randint(100, 1000))
        cross_map_lrn = torch.nn.CrossMapLRN2d(size=size, alpha=alpha, beta=beta, k=k)
        result = cross_map_lrn(input_tensor)
        return result

