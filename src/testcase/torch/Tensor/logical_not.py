import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.logical_not)
class TorchTensorLogicalUnotTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_logical_not_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]

        input_tensor = torch.randint(0, 2, input_size)  # Generate random tensor with 0 and 1
        result = input_tensor.logical_not()
        return result
