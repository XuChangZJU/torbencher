
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.rrelu)
class RReLUTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_rrelu_correctness(self):
        input_data = torch.randn(10, 10)
        lower = random.uniform(0.0, 1.0)
        upper = random.uniform(lower, 1.0)
        result = torch.nn.functional.rrelu(input_data, lower, upper)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_rrelu_large_scale(self):
        input_data = torch.randn(1000, 1000)
        lower = random.uniform(0.0, 1.0)
        upper = random.uniform(lower, 1.0)
        result = torch.nn.functional.rrelu(input_data, lower, upper)
        return result

