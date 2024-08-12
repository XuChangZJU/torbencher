import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

def shuffle(lst):
    return sorted(lst, key=lambda x: random.random())
@test_api(torch.Tensor.permute)
class TorchTensorPermuteTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_permute_correctness(self):
        # Random dimension for the tensor
        dim = random.randint(2, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Random input size
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Generate a random tensor with the specified input size
        tensor = torch.randn(input_size)
        # Generate a random permutation of the dimensions
        # dims = torch.randperm(dim).tolist()  # Generate a valid permutation of dimensions
        # Permute the dimensions of the tensor
        dims = list(range(dim))
        shuffle(dims)  # Generate valid dims by shuffling
        dims = tuple(dims)
        result = tensor.permute(dims)
        return result
