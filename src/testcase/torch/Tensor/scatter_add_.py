import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.scatter_add_)
class TorchTensorScatteraddTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_scatter_add__correctness(self):
        # Randomly generate tensor dimension
        dim = random.randint(1, 4)
        # Randomly generate number of elements for each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Generate input size list for tensors
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Generate a random tensor as the input tensor to be scattered and added
        input_tensor = torch.randn(input_size)
        # Generate a random dimension along which to index
        dim = random.randint(0, len(input_size) - 1)
        # Generate a random index tensor, its size in dimension 'dim' should be less than or equal to input_tensor's size in dimension 'dim',
        # and its size in other dimensions should be less than or equal to both input_tensor and src_tensor's size in that dimension.
        index_size = input_size.copy()
        index_size[dim] = random.randint(1, input_size[dim])
        index_tensor = torch.randint(0, input_size[dim], index_size)
        # Generate a random tensor as the source tensor to scatter and add
        src_tensor = torch.randn(index_size)

        # Apply scatter_add_ operation
        result = input_tensor.scatter_add_(dim, index_tensor, src_tensor)

        return result
