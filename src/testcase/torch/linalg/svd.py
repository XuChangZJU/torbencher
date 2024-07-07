import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.linalg.svd)
class TorchLinalgSvdTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_svd_correctness(self):
        # Randomly generate dimensions for the matrix
        m = random.randint(2, 5)
        n = random.randint(2, 5)
        
        # Generate a random matrix of shape (m, n)
        A = torch.randn(m, n)
        
        # Perform SVD
        U, S, Vh = torch.linalg.svd(A)
        
        # Verify the decomposition
        reconstructed_A = U[:, :min(m, n)] @ torch.diag(S) @ Vh[:min(m, n), :]
        return torch.dist(A, reconstructed_A)
    
    
    
    