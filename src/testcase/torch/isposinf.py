import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.isposinf)
class TorchIsposinfTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_isposinf_correctness(self):
        # Generate random dimension for the tensor
        dim = random.randint(1, 4)
        # Generate random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Generate input size
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Generate random tensor with float type
        input_tensor = torch.randn(input_size)
        # Replace some elements with inf and -inf
        input_tensor[input_tensor > 0.5] = float('inf')
        input_tensor[input_tensor <= 0.5] = -float('inf')

        result = torch.isposinf(input_tensor)
        return result