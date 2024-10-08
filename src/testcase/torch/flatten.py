import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.flatten)
class TorchFlattenTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_flatten_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]

        input_tensor = torch.randn(input_size)
        start_dim = random.randint(0, len(input_size) - 1)  # Random start dimension
        end_dim = random.randint(start_dim, len(input_size) - 1)  # Random end dimension, ensuring end_dim >= start_dim
        result = torch.flatten(input_tensor, start_dim, end_dim)
        return result
