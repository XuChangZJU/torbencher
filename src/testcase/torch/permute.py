import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

def shuffle(lst):
    return sorted(lst, key=lambda x: random.random())
@test_api(torch.permute)
class TorchPermuteTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_permute_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]

        input_tensor = torch.randn(input_size)
        dims = list(range(dim))
        shuffle(dims)  # Generate valid dims by shuffling
        dims = tuple(dims)
        result = torch.permute(input_tensor, dims)
        return result
