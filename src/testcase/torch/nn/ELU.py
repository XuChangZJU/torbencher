
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.ELU)
class TorchELUTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_elu_correctness(self):
        alpha = random.uniform(0.1, 10.0)
        input_tensor = torch.randn(random.randint(1, 10), random.randint(1, 10))
        elu = torch.nn.ELU(alpha=alpha)
        result = elu(input_tensor)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_elu_large_scale(self):
        alpha = random.uniform(0.1, 10.0)
        input_tensor = torch.randn(random.randint(1000, 10000), random.randint(100, 1000))
        elu = torch.nn.ELU(alpha=alpha)
        result = elu(input_tensor)
        return result

