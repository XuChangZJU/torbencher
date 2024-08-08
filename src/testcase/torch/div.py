import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.div)
class TorchDivTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_div_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]

        dividend = torch.randn(input_size)
        divisor = torch.randn(input_size) + 1e-6  # Add a small value to avoid division by zero
        result = torch.div(dividend, divisor)
        return result
