
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.local_response_norm)
class LocalResponseNormTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_local_response_norm_correctness(self):
        input_data = torch.randn(10, 10, 10, 10)
        size = random.randint(1, 10)
        alpha = random.uniform(0.0, 1.0)
        beta = random.uniform(0.0, 1.0)
        k = random.uniform(0.0, 1.0)
        result = torch.nn.functional.local_response_norm(input_data, size, alpha, beta, k)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_local_response_norm_large_scale(self):
        input_data = torch.randn(100, 100, 100, 100)
        size = random.randint(10, 100)
        alpha = random.uniform(0.0, 1.0)
        beta = random.uniform(0.0, 1.0)
        k = random.uniform(0.0, 1.0)
        result = torch.nn.functional.local_response_norm(input_data, size, alpha, beta, k)
        return result

