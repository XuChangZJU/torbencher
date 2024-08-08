import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.functional.binary_cross_entropy_with_logits)
class TorchNnFunctionalBinaryUcrossUentropyUwithUlogitsTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_binary_cross_entropy_with_logits_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]

        input = torch.randn(input_size)  # Tensor of arbitrary shape as unnormalized scores
        target = torch.rand(input_size)  # Tensor of the same shape as input with values between 0 and 1
        return torch.nn.functional.binary_cross_entropy_with_logits(input, target)
