import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.diff)
class TorchDiffTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_diff_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(3,
                                                  10)  # Random number of elements each dimension (ensuring at least 3 elements for diff)
        input_size = [num_of_elements_each_dim for _ in range(dim)]

        n = random.randint(1, 3)  # A small number to compute the difference 'n' times
        compute_dim = random.randint(0, dim - 1)  # Select a valid dimension to compute the difference along

        tensor = torch.randn(input_size)
        result = torch.diff(tensor, n=n, dim=compute_dim)
        return result
