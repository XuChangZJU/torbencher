import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.diff)
class TorchTensorDiffTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_diff_correctness(self):
        # Random dimension for the tensors
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Random input size
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Generate a random tensor
        input_tensor = torch.randn(input_size)
        # Random n
        n = random.randint(1, input_size[dim - 1])  # n should be less than or equal to the size of the last dimension
        # Random dim
        dim = random.randint(-len(input_size),
                             len(input_size) - 1)  # dim should be in the range of [-len(input_size), len(input_size) - 1]
        result = input_tensor.diff(n, dim)
        return result
