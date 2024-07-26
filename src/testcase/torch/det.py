import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.det)
class TorchDetTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_det_correctness(self):
        n = random.randint(2, 5)  # Random size for the square tensor
        input_tensor = torch.randn(n, n)  # Generating a random square tensor matrix
        result = torch.det(input_tensor)
        return result
