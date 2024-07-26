import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.linalg.inv_ex)
class TorchLinalgInvexTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_linalg_inv_ex_correctness(self):
        # Generate random dimension for the square matrix
        dim = random.randint(1, 10)
        # Generate random square matrix
        input_size = [dim, dim]
        A = torch.randn(input_size)
        # Calculate inverse and info
        Ainv, info = torch.linalg.inv_ex(A)
        # Return results
        return Ainv, info
