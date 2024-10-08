import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.sort)
class TorchSortTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_sort_correctness(self):
        dim = random.randint(0, 3)  # Random dimension for the tensors, valid range for 4-d tensor is [0, 3]
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(4)]  # Generate input size for 4-d tensor
        input_tensor = torch.randn(input_size)
        result = torch.sort(input_tensor, dim)
        return result
