import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.enable_grad)
class TorchEnableUgradTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_enable_grad_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]

        x = torch.randn(input_size, requires_grad=True)
        with torch.no_grad():
            with torch.enable_grad():
                y = x * 2
        result = y.requires_grad
        return result
