import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.log10)
class TorchTensorLog10TestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_log10_correctness(self):
        # Generate random dimension and size for the tensor
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Generate a random tensor with values greater than 0
        input_tensor = torch.randn(input_size)
        input_tensor = torch.abs(input_tensor)  # Make sure all elements are positive

        result = input_tensor.log10()
        return result
