import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.trace)
class TorchTraceTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_trace_correctness(self):
        rows = random.randint(2, 5)  # Random number of rows for the matrix (minimum of 2)
        cols = rows  # Ensure matrix is square to be valid for trace

        input_matrix = torch.randn(rows, cols)  # Generate a random square matrix
        result = torch.trace(input_matrix)
        return result
