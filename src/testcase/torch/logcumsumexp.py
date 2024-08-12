import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.logcumsumexp)
class TorchLogcumsumexpTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_logcumsumexp_correctness(self):
        dim = random.randint(0, 3)  # Random dimension for the operation
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in
                      range(dim + 1)]  # Generate input_size, dim+1 make sure the dimension is valid

        input_tensor = torch.randn(input_size)
        result = torch.logcumsumexp(input_tensor, dim)
        return result
