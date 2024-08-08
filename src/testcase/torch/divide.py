import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.divide)
class TorchDivideTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_divide_correctness(self):
        # Generate random dimension and size for the tensors
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Generate random tensors
        dividend = torch.randn(input_size)
        divisor = torch.randn(input_size) + 1e-6  # Add a small value to avoid division by zero

        # Perform element-wise division
        result = torch.divide(dividend, divisor)
        return result
