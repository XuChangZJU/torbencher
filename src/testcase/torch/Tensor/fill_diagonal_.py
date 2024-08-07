import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.fill_diagonal_)
class TorchTensorFillUdiagonalUTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_fill_diagonal__correctness(self):
        # Randomly generate the dimension of the tensor (at least 2)
        dim = random.randint(2, 4)
        # Randomly generate the size of each dimension (all dimensions must be of equal length)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Create a random tensor
        tensor = torch.randn(input_size)
        # Generate a random fill value
        fill_value = random.uniform(0.1, 10.0)
        # Apply the fill_diagonal_ operation
        result = tensor.fill_diagonal_(fill_value)

        return result
