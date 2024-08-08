import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.randint_like)
class TorchRandintUlikeTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_randint_like_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]
        input_tensor = torch.randn(input_size)
        low = random.randint(-10, 10)  # Random low value between -10 and 10
        high = random.randint(low + 1, 20)  # Random high value between low+1 and 20
        result = torch.randint_like(input_tensor, low, high)
        return result
