import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.corrcoef)
class TorchCorrcoefTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_corrcoef_correctness(self):
        num_variables = random.randint(1, 5)  # Random number of variables (rows in the matrix)
        num_observations = random.randint(2,
                                          10)  # Random number of observations (columns in the matrix), at least 2 to compute correlation

        # Create a random tensor with the generated size
        input_matrix = torch.randn(num_variables, num_observations)

        # Compute the correlation coefficient matrix
        result = torch.corrcoef(input_matrix)
        return result
