import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.masked_fill)
class TorchTensorMaskedUfillTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_masked_fill_correctness(self):
        # Define the dimension of the tensor
        dim = random.randint(1, 4)
        # Define the number of elements in each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Create the input size list
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Generate a random tensor with the specified input size
        input_tensor = torch.randn(input_size)
        # Generate a random boolean mask with the same size as the input tensor
        mask_tensor = torch.randint(0, 2, input_size, dtype=torch.bool)
        # Generate a random scalar value to fill the masked elements
        value = random.uniform(0.1, 10.0)

        # Apply the masked_fill operation
        result = input_tensor.masked_fill(mask_tensor, value)

        # Return the result tensor
        return result
