import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.row_stack)
class TorchRowUstackTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_row_stack_correctness(self):
        num_of_tensors = random.randint(2, 5)  # Random number of tensors to stack
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random num of elements in each dimension

        tensors = []
        for _ in range(num_of_tensors):
            input_size = [num_of_elements_each_dim for _ in range(dim)]
            tensors.append(torch.randn(input_size))

        result = torch.row_stack(tensors)
        return result
