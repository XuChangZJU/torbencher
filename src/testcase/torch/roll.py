import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.roll)
class TorchRollTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_roll_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensor
        num_of_elements_each_dim = random.randint(2, 5)  # Each dimension should have at least 2 elements
        tensor_size = [num_of_elements_each_dim for _ in range(dim)]

        tensor = torch.randn(tensor_size)
        # Generate shifts as a tuple if dim > 1, otherwise as a single int
        if dim > 1:
            shifts = tuple(random.randint(-num_of_elements_each_dim, num_of_elements_each_dim) for _ in range(dim))
            dims = tuple(range(dim))  # Use all dimensions for rolling
            result = torch.roll(tensor, shifts, dims)
        else:
            shifts = random.randint(-num_of_elements_each_dim, num_of_elements_each_dim)
            result = torch.roll(tensor, shifts)
        return result
