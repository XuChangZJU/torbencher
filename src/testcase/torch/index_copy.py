import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.index_copy)
class TorchIndexUcopyTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_index_copy_correctness(self):
        # Randomly generate tensor dimensions
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for _ in range(dim)]

        # Create input tensor
        input_tensor = torch.randn(input_size)

        # Randomly select a dimension to copy along
        dim_to_copy = random.randint(0, len(input_size) - 1)

        # Generate random indices to copy to
        index_size = random.randint(1, input_size[dim_to_copy])
        indices = torch.randint(0, input_size[dim_to_copy], (index_size,))

        # Create source tensor with matching size along the copy dimension
        source_size = input_size.copy()
        source_size[dim_to_copy] = index_size
        source_tensor = torch.randn(source_size)

        # Perform index_copy operation
        result = torch.index_copy(input_tensor, dim_to_copy, indices, source_tensor)
        return result
