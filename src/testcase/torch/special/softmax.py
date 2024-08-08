import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.special.softmax)
class TorchSpecialSoftmaxTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_softmax_correctness(self):
        # Random dimension for the tensor
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for _ in range(dim)]

        # Generate random tensor with the specified size
        input_tensor = torch.randn(input_size)
        # Randomly select a dimension along which softmax will be computed
        softmax_dim = random.randint(0, dim - 1)

        # Compute softmax
        result = torch.special.softmax(input_tensor, softmax_dim)
        return result
