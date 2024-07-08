import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.linalg.lu_factor)
class TorchLinalgLufactorTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_lu_factor_correctness(self):
        # Randomly generate dimensions for the matrix
        batch_dim = random.randint(0, 3)  # Random batch dimensions (0 to 3)
        m = random.randint(2, 5)  # Random number of rows (2 to 5)
        n = random.randint(2, 5)  # Random number of columns (2 to 5)
        
        # Generate random input size
        input_size = [random.randint(1, 5) for _ in range(batch_dim)] + [m, n]
        
        # Generate a random tensor with the specified dimensions
        A = torch.randn(input_size)
        
        # Perform LU factorization
        LU, pivots = torch.linalg.lu_factor(A)
        
        return LU, pivots
    