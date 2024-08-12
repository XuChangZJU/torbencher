import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.less_)
class TorchTensorLessUTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_less__correctness(self):
        # Randomly generate input tensor size
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Generate random input tensors
        input_tensor = torch.randn(input_size)
        other_tensor = torch.randn(input_size)

        # Perform the in-place less operation
        input_tensor.less_(other_tensor)

        return input_tensor  # Return the modified tensor to observe the in-place effect
