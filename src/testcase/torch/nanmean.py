import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nanmean)
class TorchNanmeanTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_nanmean_correctness(self):
        dim = random.randint(0, 3)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim + 1)]  # Generate input_size

        # Generate random tensor with nan values
        input_tensor = torch.randn(input_size)
        mask = torch.rand(input_size) < 0.2  # Randomly set 20% elements to NaN
        input_tensor[mask] = float('nan')

        result = torch.nanmean(input_tensor, dim)
        return result
