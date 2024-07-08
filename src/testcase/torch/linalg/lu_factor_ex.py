import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.linalg.lu_factor_ex)
class TorchLinalgLufactorexTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_lu_factor_ex_correctness(self):
        # Randomly generate dimensions for the tensor
        batch_dim = random.randint(0, 3)  # Random number of batch dimensions
        m = random.randint(1, 5)  # Random number of rows
        n = random.randint(1, 5)  # Random number of columns
    
        # Generate random input size
        input_size = [random.randint(1, 5) for _ in range(batch_dim)] + [m, n]
    
        # Create a random tensor with the generated size
        A = torch.randn(input_size)
    
        # Perform LU factorization
        LU, pivots, info = torch.linalg.lu_factor_ex(A)
    
        return LU, pivots, info
    