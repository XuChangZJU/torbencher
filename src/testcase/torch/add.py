import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.add)
class TorchAddTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_add_correctness_with_small_random_tensors(self):
        # Generate random dimension and size for the tensors
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Generate random tensors
        input_tensor = torch.randn(input_size)
        other_tensor = torch.randn(input_size)

        # Call torch.add
        result = torch.add(input_tensor, other_tensor)
        return result
