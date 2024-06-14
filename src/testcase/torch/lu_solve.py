import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.lu_solve)
class TorchLusolveTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_lu_solve_correctness(self):
        # Generate random sizes for the input tensors
        batch_size = random.randint(1, 3)
        m = random.randint(1, 5)
        k = random.randint(1, 5)
    
        # Generate random input tensors
        A = torch.randn(batch_size, m, m)
        b = torch.randn(batch_size, m, k)
    
        # Calculate LU factorization
        LU, pivots = torch.linalg.lu_factor(A)
    
        # Calculate the solution using lu_solve
        x = torch.lu_solve(b, LU, pivots)
        
        # Return the solution
        return x
    
    
    
    
    
    
    