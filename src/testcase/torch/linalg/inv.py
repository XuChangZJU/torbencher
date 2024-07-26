import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.linalg.inv)
class TorchLinalgInvTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_linalg_inv_correctness(self):
        # Generate a random square matrix that is invertible
        n = random.randint(1, 5)
        A = torch.randn(n, n)
        while torch.linalg.det(A) == 0:  # Ensure the matrix is invertible
            A = torch.randn(n, n)
        Ainv = torch.linalg.inv(A)
        return Ainv
