import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.triu_indices)
class TorchTriuUindicesTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_triu_indices_correctness(self):
        row = random.randint(1, 10)  # Random number of rows
        col = random.randint(1, 10)  # Random number of columns
        offset = random.randint(-row + 1, col - 1)  # Random offset within valid range
        result = torch.triu_indices(row, col, offset)
        return result
