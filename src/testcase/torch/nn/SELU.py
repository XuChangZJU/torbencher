
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.SELU)
class TorchSELUTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_selu_correctness(self):
        input_tensor = torch.randn(random.randint(1, 10), random.randint(1, 10))
        selu = torch.nn.SELU()
        result = selu(input_tensor)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_selu_large_scale(self):
        input_tensor = torch.randn(random.randint(1000, 10000), random.randint(100, 1000))
        selu = torch.nn.SELU()
        result = selu(input_tensor)
        return result

