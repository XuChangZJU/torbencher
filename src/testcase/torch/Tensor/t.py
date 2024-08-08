import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.t)
class TorchTensorTTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_tensor_t_correctness(self):
        # Randomly generate dimensions for a 2D tensor
        rows = random.randint(1, 5)
        cols = random.randint(1, 5)

        # Create a random 2D tensor
        tensor = torch.randn(rows, cols)

        # Transpose the tensor
        result = tensor.t()

        return result
