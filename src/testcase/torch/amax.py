import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.amax)
class TorchAmaxTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_amax_correctness(self):
        dim = random.randint(0, 3)  # Random dimension to reduce along
        keepdim = random.choice([True, False])  # Randomly choose whether to keep the reduced dimension
        input_tensor = torch.randn(2, 3, 4, 5)  # Random input tensor
        result = torch.amax(input_tensor, dim, keepdim)
        return result
