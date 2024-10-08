import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.cumsum)
class TorchCumsumTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_cumsum_correctness(self):
        dim = random.randint(0, 3)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim + 1)]

        input_tensor = torch.randn(input_size)
        dim = random.randint(0, len(input_tensor.size()) - 1)  # Random dim value
        result = torch.cumsum(input_tensor, dim)
        return result
