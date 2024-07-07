import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.linalg.eig)
class TorchLinalgEigTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_linalg_eig_correctness(self):
        # Random dimension for the square matrix
        n = random.randint(2, 5)
        
        # Randomly generate a square matrix of size (n, n) with complex values
        A = torch.randn(n, n, dtype=torch.complex128)
        
        # Compute the eigenvalue decomposition
        eigenvalues, eigenvectors = torch.linalg.eig(A)
        
        # Verify the decomposition: A should be approximately equal to V @ diag(L) @ V^-1
        reconstructed_A = eigenvectors @ torch.diag(eigenvalues) @ torch.linalg.inv(eigenvectors)
        return torch.dist(reconstructed_A, A)
    
    
    
    