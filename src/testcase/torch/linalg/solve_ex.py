import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.linalg.solve_ex)
class TorchLinalgSolveexTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_linalg_solve_ex_correctness(self):
        # Define the dimension of the matrix
        dim = random.randint(1, 10)
        # Generate random matrix A
        A = torch.randn(dim, dim)
        # Make sure A is invertible
        while torch.linalg.det(A) == 0:
            A = torch.randn(dim, dim)
        # Generate random matrix B
        B = torch.randn(dim, dim)
        # Calculate the solution and info
        result, info = torch.linalg.solve_ex(A, B)
        return result
    