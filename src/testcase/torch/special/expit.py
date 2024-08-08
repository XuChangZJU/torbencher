import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.special.expit)
class TorchSpecialExpitTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_expit_correctness(self):
        # Random dimension for the tensor
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Random input size
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Generate random input tensor
        input_tensor = torch.randn(input_size)
        # Calculate expit
        result = torch.special.expit(input_tensor)
        return result
