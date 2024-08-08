import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.linalg.qr)
class TorchLinalgQrTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_torch_linalg_qr_correctness(self):
        # Create random matrix A with shape (*, m, n)
        dim = random.randint(1, 4)
        m = random.randint(1, 5)
        n = random.randint(1, 5)
        input_size = [random.randint(1, 5) for _ in range(dim - 2)] + [m, n]
        A = torch.randn(input_size)
        # Calculate QR decomposition
        Q, R = torch.linalg.qr(A)
        # Return Q and R
        return Q, R
