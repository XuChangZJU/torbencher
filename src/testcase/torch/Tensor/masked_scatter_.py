import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.masked_scatter_)
class TorchTensorMaskedUscatterUTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_masked_scatter_correctness(self):
        # Define the dimension of the tensors
        dim = random.randint(1, 4)
        # Define the number of elements in each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Create the input size list
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Generate a random tensor
        input = torch.randn(input_size)
        # Generate a random boolean mask with the same shape as the input tensor
        mask = torch.randint(0, 2, input_size, dtype=torch.bool)
        # Generate a random source tensor with at least as many elements as the number of ones in the mask
        source_size = input_size[:]
        source_size[random.randint(0, len(input_size) - 1)] = mask.sum().item()
        source = torch.randn(source_size)
        result = input.masked_scatter_(mask, source)
        return result
