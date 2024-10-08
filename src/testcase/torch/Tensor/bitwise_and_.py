import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.bitwise_and_)
class TorchTensorBitwiseUandUTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_bitwise_and__correctness(self):
        # Define the dimension and size of the tensors randomly
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Generate random tensors
        tensor = torch.randint(0, 10, input_size)  # Generate integers for bitwise operations
        other = torch.randint(0, 10, input_size)  # Generate integers for bitwise operations

        # Perform in-place bitwise AND operation
        tensor.bitwise_and_(other)

        return tensor
