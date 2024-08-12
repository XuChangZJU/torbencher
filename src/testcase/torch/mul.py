import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.mul)
class TorchMulTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_mul_correctness_tensor_tensor(self):
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        input = torch.randn(input_size)
        other = torch.randn(input_size)
        result = torch.mul(input, other)
        return result
