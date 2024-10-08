import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.softmax)
class TorchTensorSoftmaxTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_softmax_correctness(self):
        # Generate random dimension for the tensor
        dim = random.randint(1, 4)
        # Generate random number of elements for each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Generate input size
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Generate random tensor
        tensor = torch.randn(input_size)
        # Generate random dim
        dim = random.randint(-len(input_size), len(input_size) - 1)
        # Calculate softmax
        result = tensor.softmax(dim)
        return result
