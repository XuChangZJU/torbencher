import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.linalg.ldl_factor)
class TorchLinalgLdlUfactorTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_torch_linalg_ldl_factor_correctness(self):
        # Define the dimension of the matrix
        dim = random.randint(1, 10)
        # Generate a random square matrix A
        A = torch.randn(dim, dim)
        # Make the matrix symmetric
        A = A @ A.mT
        # Calculate the LDL factorization
        LD, pivots = torch.linalg.ldl_factor(A)
        # Return the results
        return LD, pivots
