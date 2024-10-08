import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.special.round)
class TorchSpecialRoundTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_special_round_correctness(self):
        # Generate random dimension for the tensor
        dim = random.randint(1, 4)
        # Generate random number of elements for each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Create input size list
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Generate random tensor data
        input_tensor = torch.randn(input_size)
        # Call torch.special.round
        result = torch.special.round(input_tensor)
        return result
