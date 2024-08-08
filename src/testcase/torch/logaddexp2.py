import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.logaddexp2)
class TorchLogaddexp2TestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_logaddexp2_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]

        input = torch.randn(input_size)
        other = torch.randn(input_size)
        result = torch.logaddexp2(input, other)
        return result
