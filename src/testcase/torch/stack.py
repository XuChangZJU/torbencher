import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.stack)
class TorchStackTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_stack_correctness(self):
        num_of_tensors = random.randint(2, 5)  # Number of tensors to concatenate
        tensor_dim = random.randint(1, 4)  # Random dimension for each tensor
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements in each dimension
        dim = random.randint(0, tensor_dim)  # Dimension to insert (must be within the range of tensor_dim)

        input_size = [num_of_elements_each_dim for _ in range(tensor_dim)]  # Input size for each tensor
        tensors = [torch.randn(input_size) for _ in range(num_of_tensors)]  # Sequence of tensors to concatenate

        result = torch.stack(tensors, dim)
        return result
