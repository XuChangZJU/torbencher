import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.nan_to_num_)
class TorchTensorNanUtoUnumUTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_nan_to_num__correctness(self):
        # Randomly generate tensor dimension and size
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Create a random tensor with NaN, Inf, and -Inf values
        tensor = torch.randn(input_size)
        tensor[tensor > 0.9] = float('inf')
        tensor[tensor < -0.9] = float('-inf')
        tensor[torch.rand(input_size) > 0.9] = float('nan')

        # Apply nan_to_num_ in-place
        tensor.nan_to_num_()

        return tensor
