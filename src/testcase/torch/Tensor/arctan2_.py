import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.arctan2_)
class TorchTensorArctan2UTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_arctan2__correctness(self):
        # Randomly generate input tensor size
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Generate random input tensors
        input_tensor = torch.randn(input_size)
        other_tensor = torch.randn(input_size)  # The size should be same as input

        # Apply arctan2_ operation in-place
        input_tensor.arctan2_(other_tensor)

        return input_tensor
