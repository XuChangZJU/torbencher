import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.roll)
class TorchTensorRollTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_roll_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensor
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for _ in range(dim)]  # Generate input size for the tensor

        tensor = torch.randn(input_size)  # Generate a random tensor
        shifts = random.randint(-num_of_elements_each_dim,
                                num_of_elements_each_dim)  # Random shift value within valid range
        dims = random.randint(0, dim - 1)  # Random dimension to apply the roll

        result = tensor.roll(shifts, dims)  # Apply the roll operation
        return result
