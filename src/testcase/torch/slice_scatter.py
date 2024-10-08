import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.slice_scatter)
class TorchSliceUscatterTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_slice_scatter_correctness(self):
        # Random dimension and size for the input tensor
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(2, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Create input tensor
        input_tensor = torch.randn(input_size)

        # Select a random dimension to apply slice_scatter
        slice_dim = random.randint(0, dim - 1)

        # Random start and end indices
        start_idx = random.randint(0, num_of_elements_each_dim - 1)
        end_idx = random.randint(start_idx + 1, num_of_elements_each_dim)

        # Create src tensor which has the same shape as the slice of input_tensor
        src_size = input_size[:]
        src_size[slice_dim] = end_idx - start_idx
        src_tensor = torch.randn(src_size)

        # Perform the slice_scatter operation; step is left as 1 (default)
        result_tensor = torch.slice_scatter(input_tensor, src_tensor, slice_dim, start_idx, end_idx)

        return result_tensor
