import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch._foreach_reciprocal)
class TorchUforeachUreciprocalTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_foreach_reciprocal_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]
        num_tensors = random.randint(1, 5)  # Random number of tensors in the list
        input_tensors = [torch.randn(input_size) for _ in range(num_tensors)]  # List of random tensors
        result = torch._foreach_reciprocal(input_tensors)
        return result
