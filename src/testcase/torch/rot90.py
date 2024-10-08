import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.rot90)
class TorchRot90TestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_rot90_correctness(self):
        dim = random.randint(2, 4)  # Random dimension for the tensors, at least 2 dimension
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]

        input_tensor = torch.randn(input_size)
        k = random.randint(0, 5)  # Random k value
        # dims_value = random.sample(range(0, dim), 2)  # Randomly select two different values from 0 to dim - 1
        dims = random.randint(0, dim-1), random.randint(0, dim-1)
        while dims[0] == dims[1]:
            dims = random.randint(0, dim-1), random.randint(0, dim-1)

        result = torch.rot90(input_tensor, k, dims)
        return result
