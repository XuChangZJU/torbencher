import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.exponential_)
class TorchTensorExponentialUTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_exponential__correctness(self):
        dim = 4  # Random dimension for the tensors
        num_of_elements_each_dim = 5  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]

        input_tensor = torch.randn(input_size)
        lambd = random.uniform(0.1, 10.0)  # Random lambd value between 0.1 and 10.0
        result = input_tensor.exponential_(lambd)
        return result.shape
