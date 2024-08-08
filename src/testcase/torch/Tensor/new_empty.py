import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.new_empty)
class TorchTensorNewUemptyTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_new_empty_correctness(self):
        # Generate random dimension for the tensor
        dim = random.randint(1, 4)
        # Generate random number of elements for each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Create input size list
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Create a tensor with random data
        tensor = torch.randn(input_size)
        # Create a new empty tensor with the same dtype and device as the input tensor
        result = tensor.new_empty(input_size)
        # Return the result tensor
        return result
