import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.cummin)
class TorchCumminTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_cummin_correctness(self):
        dim = random.randint(0, 3)  # Random dimension for the operation
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim + 1)]

        input_tensor = torch.randn(input_size)
        result = torch.cummin(input_tensor, dim)
        return result
