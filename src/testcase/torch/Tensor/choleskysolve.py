import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.choleskysolve)
class TorchTensorCholeskysolveTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_cholesky_solve_correctness(self):
    # Random dimension for the square matrix
    dim = random.randint(2, 5)
    
    # Random number of elements for the square matrix
    num_of_elements = random.randint(2, 5)
    
    # Generate a random positive-definite matrix for cholesky decomposition
    A = torch.randn(dim, dim)
    A = torch.mm(A, A.t()) + torch.eye(dim) * 1e-5  # Make it positive-definite
    
    # Perform Cholesky decomposition
    L = torch.cholesky(A, upper=False)
    
    # Generate a random tensor for the right-hand side
    B = torch.randn(dim, num_of_elements)
    
    # Solve the linear system using cholesky_solve
    result = torch.cholesky_solve(B, L)
    
    return result
