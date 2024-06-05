
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.celu)
class CeluTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_celu_correctness(self):
        input_data = torch.randn(10, 10)
        alpha = random.uniform(0.0, 10.0)
        result = torch.nn.functional.celu(input_data, alpha)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_celu_large_scale(self):
        input_data = torch.randn(1000, 1000)
        alpha = random.uniform(0.0, 10.0)
        result = torch.nn.functional.celu(input_data, alpha)
        return result

