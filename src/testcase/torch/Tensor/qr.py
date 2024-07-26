import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.qr)
class TorchTensorQrTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_qr_correctness(self):
        # Randomly generate dimensions for the tensor
        rows = random.randint(2, 5)  # Number of rows (must be at least 2 for QR decomposition)
        cols = random.randint(2, 5)  # Number of columns (must be at least 2 for QR decomposition)

        # Generate a random tensor with the specified dimensions
        tensor = torch.randn(rows, cols)

        # Perform QR decomposition
        Q, R = tensor.qr()

        # Return the Q and R matrices
        return Q, R
