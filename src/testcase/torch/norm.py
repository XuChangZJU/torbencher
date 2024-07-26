import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.norm)
class TorchNormTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_norm_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]

        input_tensor = torch.randn(input_size)
        p = random.choice([2, float('inf'), -float('inf'), 1, -1, 0])  # Randomly choose a valid value for p
        dim = random.randint(0, len(input_size) - 1)  # Randomly choose a valid dimension
        result = torch.norm(input_tensor, p, dim)
        return result
