import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.logdet)
class TorchLogdetTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_logdet_correctness(self):
        # Generate a random square matrix
        dim = random.randint(1, 10)  # Random dimension for the matrix
        input_size = [dim, dim]

        # Generate a random tensor of the specified size
        input_tensor = torch.randn(input_size)

        # Calculate the log determinant using torch.logdet
        result = torch.logdet(input_tensor)
        return result
