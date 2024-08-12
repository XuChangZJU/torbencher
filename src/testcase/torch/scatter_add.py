import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.scatter_add)
class TorchScatterUaddTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_scatter_add_correctness(self):
        # Define the dimension of the tensor
        dim = random.randint(0, 3)  # dim (int): the axis along which to index
        # Define the size of the tensor
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim + 1)]
        # Generate random input tensor
        input = torch.randn(input_size)
        # Generate random index tensor
        index_size = input_size.copy()
        index_size[dim] = random.randint(1, input_size[dim])
        index = torch.randint(0, input_size[dim], index_size)
        # Generate random source tensor with the same size as index
        src = torch.randn(index_size)
        result = torch.scatter_add(input, dim, index, src)
        return result
