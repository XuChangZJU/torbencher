import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.xlogy_)
class TorchTensorXlogyUTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_xlogy__correctness(self):
        # Define the dimension and size of the input tensors randomly
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Generate random tensors
        input_tensor = torch.randn(input_size)  # Tensor to be modified in-place
        other_tensor = torch.randn(input_size)  # Tensor used for element-wise xlogy calculation

        # Perform in-place xlogy operation
        input_tensor.xlogy_(other_tensor)

        return input_tensor  # Return the modified tensor to observe the effect of xlogy_
