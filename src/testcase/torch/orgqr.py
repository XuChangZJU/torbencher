import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.orgqr)
class TorchOrgqrTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_orgqr_correctness(self):
        # Generate random input size
        dim = random.randint(1, 4)
        m = random.randint(1, 5)
        n = random.randint(1, min(m, 5))  # n <= m for valid QR decomposition
        input_size = [m, n]

        # Generate random input tensor
        input_tensor = torch.randn(input_size)

        # Perform QR decomposition to obtain 'tau'
        q, r = torch.linalg.qr(input_tensor)
        tau = torch.diag(r)

        # Calculate and return the result of orgqr
