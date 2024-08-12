import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.isfinite)
class TorchIsfiniteTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_isfinite_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]

        input_tensor = torch.tensor([float('inf'), float('-inf'), float('nan'), 1, 2,
                                     3])  # Create a tensor with infinity, negative infinity, NaN, and finite values
        result = torch.isfinite(input_tensor)
        return result
