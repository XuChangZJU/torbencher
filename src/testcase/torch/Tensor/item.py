import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.item)
class TorchTensorItemTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_item_correctness(self):
        # Generate random dimension for the tensor
        dim = random.randint(1, 4)
        # Generate random number of elements for each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Create input size list
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Make sure the tensor has only one element
        input_size = [1]
        # Generate a random tensor with the specified size
        tensor = torch.randn(input_size)
        # Get the item from the tensor
        result = tensor.item()
        return result
