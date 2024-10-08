import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.le)
class TorchLeTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_le_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]

        input_tensor = torch.randn(input_size)
        # Generate other tensor with the same shape as input_tensor to ensure broadcasting works
        other_tensor = torch.randn(input_size)
        result = torch.le(input_tensor, other_tensor)
        return result
