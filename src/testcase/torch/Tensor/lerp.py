import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.lerp)
class TorchTensorLerpTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_lerp_correctness(self):
        # Generate random dimension for the tensors
        dim = random.randint(1, 4)
        # Generate random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Generate input_size
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Generate random tensors with the same size
        start = torch.randn(input_size)
        end = torch.randn(input_size)
        # weight should be a tensor or a float number
        weight = random.uniform(0.1, 10.0)
        result = start.lerp(end, weight)
        return result
