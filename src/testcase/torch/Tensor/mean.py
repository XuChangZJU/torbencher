import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.mean)
class TorchTensorMeanTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_mean_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]

        input_tensor = torch.randn(input_size)
        dim_to_reduce = random.randint(0, dim - 1)  # Random valid dimension to reduce
        result = input_tensor.mean(dim_to_reduce)
        return result
