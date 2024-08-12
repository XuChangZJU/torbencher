import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.greater_equal)
class TorchGreaterUequalTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_greater_equal_correctness(self):
        # Random dimension for the tensors
        dim = random.randint(1, 4)
        # Random number of elements in each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for _ in range(dim)]

        # Generate two tensors of the same size with random values
        tensor1 = torch.randn(input_size)
        tensor2 = torch.randn(input_size)

        # Compute greater_equal comparison
        result = torch.greater_equal(tensor1, tensor2)
        return result
