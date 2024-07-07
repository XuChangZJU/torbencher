import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.linalg.solve)
class TorchLinalgSolveTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_solve_correctness(self):
        # Random dimension for the square matrix A
        n = random.randint(2, 5)
        
        # Random number of columns for matrix B
        k = random.randint(1, 5)
        
        # Generate random invertible matrix A of shape (n, n)
        A = torch.randn(n, n)
        while torch.det(A) == 0:  # Ensure A is invertible
            A = torch.randn(n, n)
        
        # Generate random matrix B of shape (n, k)
        B = torch.randn(n, k)
        
        # Solve the linear system AX = B
        X = torch.linalg.solve(A, B)
        
        # Verify the solution
        assert torch.allclose(A @ X, B, atol=1e-6), "The solution X does not satisfy AX = B"
        
        return X
    
    
    
    