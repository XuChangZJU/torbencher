import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.select_scatter)
class TorchTensorSelectUscatterTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_select_scatter_correctness(self):
        dim = random.randint(0, 3)  # Random dimension for the tensors (0 to 3 inclusive)
        num_of_elements_each_dim = random.randint(2, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for _ in range(dim + 1)]  # Create size for each dimension

        tensor = torch.randn(input_size)  # Original tensor
        src = torch.randn(input_size)  # Source tensor
        index = random.randint(0, num_of_elements_each_dim - 1)  # Random index within the valid range

        result = tensor.select(dim, index).copy_(src.select(dim, index))
        return result
