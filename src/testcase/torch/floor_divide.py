import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.floor_divide)
class TorchFloorUdivideTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_floor_divide_correctness(self):
        # Define the dimension and size of the tensors
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Generate random tensors
        dividend = torch.randn(input_size) * random.uniform(1, 10)  # dividend
        divisor = torch.randn(input_size) * random.uniform(1, 10) + 0.1  # divisor, ensure not containing 0.

        # Perform floor division
        result = torch.floor_divide(dividend, divisor)
        return result
