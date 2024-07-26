import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.cholesky_inverse)
class TorchTensorCholeskyinverseTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_cholesky_inverse_correctness(self):
        dim = random.randint(2, 4)  # Random dimension for the square matrix (must be at least 2x2 for Cholesky decomposition)
        matrix_size = [dim, dim]  # Square matrix size
    
        # Generate a random positive-definite matrix
        A = torch.randn(matrix_size)
        positive_definite_matrix = torch.mm(A, A.t()) + dim * torch.eye(dim)  # Ensure the matrix is positive-definite
    
        # Perform Cholesky decomposition
        cholesky_factor = torch.cholesky(positive_definite_matrix)
    
        # Compute the inverse using cholesky_inverse
        result = torch.cholesky_inverse(cholesky_factor)
        return result
    
    
    
    