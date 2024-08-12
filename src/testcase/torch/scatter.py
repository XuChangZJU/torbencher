import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.scatter)
class TorchScatterTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_scatter_correctness(self):
        dim = random.randint(0, 3)  # Randomly choosing a dimension along which to scatter
        num_of_elements_each_dim = random.randint(2, 5)  # Random number of elements in each dimension

        # Generate tensor shapes
        input_size = [num_of_elements_each_dim for _ in range(dim + 1)]

        # Generate input tensors
        input_tensor = torch.randn(input_size)
        src_tensor = torch.randn(input_size)
        index_tensor = torch.randint(0, input_size[dim],
                                     input_size)  # Index tensor must have values within input_size[dim]

        result = torch.scatter(input_tensor, dim, index_tensor, src_tensor)
        return result
