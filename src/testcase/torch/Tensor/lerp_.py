import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.lerp_)
class TorchTensorLerpUTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_lerp__correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]

        start_tensor = torch.randn(input_size)
        end_tensor = torch.randn(input_size)
        weight = random.uniform(0.1, 10.0)  # Random weight value between 0.1 and 10.0
        result = start_tensor.lerp_(end_tensor, weight)
        return result
