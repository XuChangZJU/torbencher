import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.unique)
class TorchUniqueTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_unique_correctness(self):
        dim = random.randint(0, 3)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]
        input_tensor = torch.randint(0, 10, input_size)  # Generate random tensor with elements between 0 and 9
        result = torch.unique(input_tensor)
        return result
