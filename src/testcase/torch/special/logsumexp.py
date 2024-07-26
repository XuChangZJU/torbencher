import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.special.logsumexp)
class TorchSpecialLogsumexpTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_logsumexp_correctness(self):
        # Random dimension for the tensor
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for _ in range(dim)]

        # Generate random tensor
        input_tensor = torch.randn(input_size)
        # Randomly select a dimension to perform logsumexp
        dim_to_reduce = random.randint(0, dim - 1)

        # Perform logsumexp operation
        result = torch.special.logsumexp(input_tensor, dim_to_reduce)
        return result
