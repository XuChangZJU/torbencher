import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.linalg.ldl_solve)
class TorchLinalgLdlsolveTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_torch_linalg_ldl_solve_correctness(self):
        # Define the dimension of the matrix
        dim = random.randint(1, 10)
        # Define the batch size
        batch_size = random.randint(1, 5)
        # Generate a random batch of square matrices
        A = torch.randn(batch_size, dim, dim)
        # Make the matrices symmetric positive definite
        A = A @ A.mT 
        # Generate a random right-hand side tensor
        B = torch.randn(batch_size, dim, dim)
        # Perform LDL factorization
        LD, pivots, info = torch.linalg.ldl_factor_ex(A)
        # Solve the linear system using ldl_solve
        X = torch.linalg.ldl_solve(LD, pivots, B)
        # Return the solution
        return X 
    