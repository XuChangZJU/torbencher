import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.unflatten)
class TorchUnflattenTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_unflatten_correctness(self):
        # Generate random dimensions for the input tensor
        dim1 = random.randint(1, 5)
        dim2 = random.randint(1, 5)
        dim3 = random.randint(1, 5)

        # Creating a random input tensor of shape (dim1, dim2, dim3)
        input_tensor = torch.randn(dim1, dim2, dim3)

        # Select a random dimension to unflatten, must be within input shape limits
        dim_to_unflatten = random.randint(0, len(input_tensor.shape) - 1)

        # Generate valid sizes for unflattening the chosen dimension
        # Ensure that product of sizes equals the size of the selected dimension
        selected_dim_size = input_tensor.shape[dim_to_unflatten]

        # Generate random sizes that multiply to selected_dim_size
        size1 = random.randint(1, selected_dim_size)
        while selected_dim_size % size1 != 0:
            size1 = random.randint(1, selected_dim_size)
        size2 = selected_dim_size // size1

        sizes = (size1, size2)

        # Perform the unflatten operation
        result_tensor = torch.unflatten(input_tensor, dim_to_unflatten, sizes)
        return result_tensor
