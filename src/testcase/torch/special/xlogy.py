import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.special.xlogy)
class TorchSpecialXlogyTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_xlogy_correctness(self):
        # Random dimension for the tensors
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Random input size
        input_size = [num_of_elements_each_dim for _ in range(dim)]

        # Generate random tensors
        input_tensor = torch.randn(input_size)
        other_tensor = torch.randn(input_size)
        other_tensor = torch.abs(other_tensor)

        # Calculate the result using torch.special.xlogy
        result = torch.special.xlogy(input_tensor, other_tensor)
        return result
