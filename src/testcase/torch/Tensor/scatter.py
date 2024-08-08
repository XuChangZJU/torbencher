import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.scatter)
class TorchTensorScatterTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_scatter_correctness(self):
        dim = random.randint(0, 3)  # Random dimension for the scatter operation
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for _ in range(dim + 1)]  # Ensure input size matches the dimension

        # Generate random tensors for input, index, and source
        input_tensor = torch.randn(input_size)
        index_tensor = torch.randint(0, num_of_elements_each_dim, input_size)
        src_tensor = torch.randn(input_size)

        # Perform scatter operation
        result = input_tensor.scatter(dim, index_tensor, src_tensor)
        return result
