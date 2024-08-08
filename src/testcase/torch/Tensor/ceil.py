import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.ceil)
class TorchTensorCeilTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_ceil_correctness(self):
        """Test correctness of torch.Tensor.ceil with random parameters.
        """
        dim = random.randint(1, 4)  # Random dimension for the tensor
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]

        tensor = torch.randn(input_size)  # Random tensor
        result = tensor.ceil()
        return result
