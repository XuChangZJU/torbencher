import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.logaddexp2)
class TorchTensorLogaddexp2TestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_logaddexp2_correctness(self):
        # Define the dimension and size of the input tensors randomly
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Generate two random tensors of the same size
        input_tensor = torch.randn(input_size)
        other_tensor = torch.randn(input_size)
        # Calculate the element-wise logaddexp2 of the two tensors
        result = input_tensor.logaddexp2(other_tensor)
        # Return the result tensor
        return result
