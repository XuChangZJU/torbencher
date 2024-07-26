import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.cov)
class TorchTensorCovTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_cov_correctness(self):
        # Randomly generate the number of rows and columns for the input tensor
        num_rows = random.randint(2, 5)  # At least 2 rows to compute covariance
        num_cols = random.randint(2, 5)  # At least 2 columns to compute covariance

        # Generate a random tensor with the specified size
        input_tensor = torch.randn(num_rows, num_cols)

        # Compute the covariance matrix
        result = input_tensor.cov()

        return result
