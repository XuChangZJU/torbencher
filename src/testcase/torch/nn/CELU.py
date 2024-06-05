
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.CELU)
class TorchCELUTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_celu_correctness(self):
        alpha = random.uniform(0.1, 10.0)
        input_tensor = torch.randn(random.randint(1, 10), random.randint(1, 10))
        celu = torch.nn.CELU(alpha=alpha)
        result = celu(input_tensor)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_celu_large_scale(self):
        alpha = random.uniform(0.1, 10.0)
        input_tensor = torch.randn(random.randint(1000, 10000), random.randint(100, 1000))
        celu = torch.nn.CELU(alpha=alpha)
        result = celu(input_tensor)
        return result

