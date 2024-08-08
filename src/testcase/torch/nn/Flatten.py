import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.Flatten)
class TorchNnFlattenTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_flatten_correctness(self):
        # Random dimension for the tensors
        dim = random.randint(2, 5)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Random start_dim and end_dim
        start_dim = random.randint(0, dim - 2)  # start_dim should be in range [0, dim - 2]
        end_dim = random.randint(start_dim + 1, dim - 1)  # end_dim should be in range [start_dim + 1, dim - 1]

        input_tensor = torch.randn(input_size)
        flatten = torch.nn.Flatten(start_dim, end_dim)
        result = flatten(input_tensor)
        return result
