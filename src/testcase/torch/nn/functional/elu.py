
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.elu)
class EluTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_elu_correctness(self):
        input_data = torch.randn(10, 10)
        alpha = random.uniform(0.0, 10.0)
        result = torch.nn.functional.elu(input_data, alpha)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_elu_large_scale(self):
        input_data = torch.randn(1000, 1000)
        alpha = random.uniform(0.0, 10.0)
        result = torch.nn.functional.elu(input_data, alpha)
        return result

