import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.arctanh)
class TorchArctanhTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_arctanh_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensor
        num_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_elements_each_dim for _ in range(dim)]

        tensor = torch.randn(input_size).clamp(-0.99, 0.99)  # Ensure values are in the domain of arctanh (-1, 1)
        result = torch.arctanh(tensor)
        return result
