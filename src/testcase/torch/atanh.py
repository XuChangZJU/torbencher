import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.atanh)
class TorchAtanhTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_atanh_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Generate random tensor data within the valid range (-1, 1)
        input_tensor = torch.randn(input_size).clamp(min=-0.999,max=0.999)
        result = torch.atanh(input_tensor)
        return result
