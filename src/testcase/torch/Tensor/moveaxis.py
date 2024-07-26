import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.moveaxis)
class TorchTensorMoveaxisTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_moveaxis_correctness(self):
        # Define the dimension of the tensor
        dim = random.randint(2, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Generate the input size
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Generate a random tensor with the specified input size
        input_tensor = torch.randn(input_size)
        # Generate random source and destination within the valid range
        source = random.randint(0, dim - 1)
        destination = random.randint(0, dim - 1)
        # Apply the moveaxis operation
        result = input_tensor.moveaxis(source, destination)
        return result
