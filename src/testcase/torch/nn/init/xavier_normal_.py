import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.init.xavier_normal_)
class TorchNnInitXavierUnormalUTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_xavier_normal_correctness(self):
        # Randomly generate dimensions for the tensor
        dim = random.randint(2, 4)  # Ensure at least 2 dimensions
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for _ in range(dim)]

        # Create a tensor with the generated dimensions
        tensor = torch.empty(input_size)

        # Randomly generate a gain value between 0.1 and 10.0
        gain = random.uniform(0.1, 10.0)

        # Apply Xavier normal initialization
        result = torch.nn.init.xavier_normal_(tensor, gain=gain)
        return result
