import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.mul_)
class TorchTensorMulUTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_mul__correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]

        input_tensor = torch.randn(input_size)
        value = random.uniform(0.1, 10.0)  # Random value to multiply with
        input_tensor_copy = torch.clone(input_tensor)
        result = input_tensor_copy.mul_(value)
        return result
