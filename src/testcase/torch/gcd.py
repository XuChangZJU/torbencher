import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.gcd)
class TorchGcdTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_gcd_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Generate random integer tensors with values likely to have common divisors
        tensor1 = torch.randint(low=1, high=100, size=input_size)
        tensor2 = torch.randint(low=1, high=50, size=input_size)
        result = torch.gcd(tensor1, tensor2)
        return result
