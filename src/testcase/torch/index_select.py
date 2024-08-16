import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.index_select)
class TorchIndexUselectTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_index_select_correctness(self):
        # Randomly generate the dimension of the input tensor
        dim = random.randint(1, 4)
        # Randomly generate the number of elements for each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Create the input size list for the tensor
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Generate a random tensor with the specified input size
        input_tensor = torch.randn(input_size)
        # Randomly select a dimension to index
        dim_to_index = random.randint(0, dim - 1)
        # Generate random indices to select
        # The number of indices should be less than or equal to the size of the dimension being indexed
        num_of_indices = random.randint(1, input_size[dim_to_index])
        # The indices should be within the range of the dimension being indexed
        indices = torch.randint(0, input_size[dim_to_index], (num_of_indices,))
        result = torch.index_select(input_tensor, dim_to_index, indices)
        return result
