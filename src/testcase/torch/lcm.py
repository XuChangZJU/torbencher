import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.lcm)
class TorchLcmTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_lcm_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Generate random integer tensors
        input_tensor = torch.randint(1, 10, input_size)  # Integers between 1 and 10
        other_tensor = torch.randint(1, 10, input_size)  # Integers between 1 and 10
        result = torch.lcm(input_tensor, other_tensor)
        return result
