import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.linalg.cross)
class TorchLinalgCrossTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_torch_linalg_cross_correctness(self):
        # Define the dimension for the cross product (must be 3)
        dim = 3
        # Define the size of the batch of vectors
        batch_size = random.randint(1, 4)
        # Generate random input tensors with shape (batch_size, 3)
        input_tensor = torch.randn(batch_size, dim)
        other_tensor = torch.randn(batch_size, dim)
        # Compute the cross product
        result = torch.linalg.cross(input_tensor, other_tensor)
        return result
