
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.LocalResponseNorm)
class TorchLocalResponseNormTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_localresponsenorm_correctness(self):
        size = random.randint(1, 10)
        alpha = random.uniform(0.1, 10.0)
        beta = random.uniform(0.1, 10.0)
        k = random.uniform(0.1, 10.0)
        input_tensor = torch.randn(random.randint(1, 10), random.randint(1, 10), random.randint(1, 10), random.randint(1, 10))
        local_response_norm = torch.nn.LocalResponseNorm(size=size, alpha=alpha, beta=beta, k=k)
        result = local_response_norm(input_tensor)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_localresponsenorm_large_scale(self):
        size = random.randint(100, 1000)
        alpha = random.uniform(0.1, 10.0)
        beta = random.uniform(0.1, 10.0)
        k = random.uniform(0.1, 10.0)
        input_tensor = torch.randn(random.randint(1000, 10000), random.randint(100, 1000), random.randint(100, 1000), random.randint(100, 1000))
        local_response_norm = torch.nn.LocalResponseNorm(size=size, alpha=alpha, beta=beta, k=k)
        result = local_response_norm(input_tensor)
        return result

