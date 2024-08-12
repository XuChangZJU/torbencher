import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.igammac)
class TorchIgammacTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_igammac_correctness(self):
        # Generate random input tensors
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        input_tensor = torch.rand(input_size)  # input should be positive real numbers
        other_tensor = torch.rand(input_size)  # other should be positive real numbers

        result = torch.igammac(input_tensor, other_tensor)
        return result
