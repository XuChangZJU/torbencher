import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.bitwise_and)
class TorchBitwiseUandTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_bitwise_and_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]

        input1 = torch.randint(-10, 10, input_size)  # generate random tensor with int within range [-10, 10)
        input2 = torch.randint(-10, 10, input_size)  # generate random tensor with int within range [-10, 10)
        result = torch.bitwise_and(input1, input2)
        return result
