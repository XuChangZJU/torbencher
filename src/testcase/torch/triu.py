import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.triu)
class TorchTriuTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_triu_correctness(self):
        dim = random.randint(2, 4)  # Random dimension for the tensors (at least 2 dimension is required)
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]

        input_tensor = torch.randn(input_size)
        diagonal = random.randint(-num_of_elements_each_dim + 1,
                                  num_of_elements_each_dim - 1)  # diagonal should be in range to make sure the operation is valid
        result = torch.triu(input_tensor, diagonal)
        return result
