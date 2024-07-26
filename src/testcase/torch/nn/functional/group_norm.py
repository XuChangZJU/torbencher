import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.functional.group_norm)
class TorchNnFunctionalGroupnormTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_group_norm_correctness(self):
        # Random input size
        dim = random.randint(2, 4)  # Dimension should be at least 2 for group norm
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Random input tensor
        input_tensor = torch.randn(input_size)

        # Random number of groups
        num_channels = input_size[1]  # Number of channels
        num_groups = random.randint(1, num_channels)  # num_groups should divide num_channels
        while num_channels % num_groups != 0:
            num_groups = random.randint(1, num_channels)

        # Random weight and bias
        weight = torch.randn(num_channels)
        bias = torch.randn(num_channels)

        # Apply group normalization
        result = torch.nn.functional.group_norm(input_tensor, num_groups, weight, bias)
        return result
