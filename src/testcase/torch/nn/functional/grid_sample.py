import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.functional.grid_sample)
class TorchNnFunctionalGridsampleTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_grid_sample_correctness(self):
        # Random dimension for the input tensor
        dim = random.randint(4, 5)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Generate input_size based on dim
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Generate random input tensor
        input_tensor = torch.randn(input_size)
        # Generate grid_size based on dim
        grid_size = input_size.copy()
        if dim == 4:
            grid_size[-1] = 2  # grid_size should be (N, H_out, W_out, 2)
        else:
            grid_size[-1] = 3  # grid_size should be (N, D_out, H_out, W_out, 3)
        # Generate random grid tensor
        grid = torch.randn(grid_size)
        # Make sure grid values are within [-1, 1]
        grid = grid.clamp(-1, 1)
        # Call grid_sample
        result = torch.nn.functional.grid_sample(input_tensor, grid)
        return result
