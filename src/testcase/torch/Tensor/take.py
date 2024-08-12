import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.take)
class TorchTensorTakeTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_take_correctness(self):
        # Generate random dimension for the input tensor
        dim = random.randint(1, 4)
        # Generate random number of elements for each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Create input_size list
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Generate random input tensor 
        input_tensor = torch.randn(input_size)
        # Generate random indices tensor with the same number of dimensions as input_tensor
        # The elements in indices tensor should be within the range of the input_tensor size
        indices_tensor = torch.randint(0, input_tensor.numel(), size=input_size)
        # Call torch.Tensor.take()
        result = input_tensor.take(indices_tensor)
        return result
