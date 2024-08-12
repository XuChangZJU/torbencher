import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.pow_)
class TorchTensorPowUTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_pow__correctness(self):
        # Randomly generate dimension and size of the tensor
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Generate a random tensor
        input_tensor = torch.abs(torch.randn(input_size)) + 1e-5

        # Generate a random exponent
        exponent = random.uniform(0.1, 10.0)

        # Apply pow_ operation in-place
        input_tensor.pow_(exponent)

        return input_tensor
