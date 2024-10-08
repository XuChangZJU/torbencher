import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.t_)
class TorchTensorTUTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_t__correctness(self):
        # Randomly generate dimensions for a 2D tensor
        rows = random.randint(1, 5)
        cols = random.randint(1, 5)

        # Create a random 2D tensor
        tensor = torch.randn(rows, cols)

        # Apply the in-place transpose operation
        result = tensor.t_()

        return result
