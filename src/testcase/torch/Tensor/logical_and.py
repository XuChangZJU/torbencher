import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.logical_and)
class TorchTensorLogicalUandTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_logical_and_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]

        input1 = torch.randn(input_size) > 0  # generate random tensor with element 0 or 1
        input2 = torch.randn(input_size) > 0  # generate random tensor with element 0 or 1
        result = input1.logical_and(input2)
        return result
