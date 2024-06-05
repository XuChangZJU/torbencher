
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.gumbel_softmax)
class GumbelSoftmaxTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_gumbel_softmax_correctness(self):
        input_data = torch.randn(10, 10)
        tau = random.uniform(0.0, 1.0)
        hard = random.choice([True, False])
        dim = random.randint(0, 9)
        result = torch.nn.functional.gumbel_softmax(input_data, tau, hard, dim)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_gumbel_softmax_large_scale(self):
        input_data = torch.randn(1000, 1000)
        tau = random.uniform(0.0, 1.0)
        hard = random.choice([True, False])
        dim = random.randint(0, 999)
        result = torch.nn.functional.gumbel_softmax(input_data, tau, hard, dim)
        return result

