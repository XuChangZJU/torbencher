import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.count_nonzero)
class TorchTensorCountUnonzeroTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_count_nonzero_correctness(self):
        # Random dimension for the tensors
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Generate random input size
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Generate random tensor
        input_tensor = torch.randn(input_size)
        # Calculate count_nonzero
        result = input_tensor.count_nonzero()
        return result
