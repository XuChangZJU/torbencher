import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.lobpcg)
class TorchLobpcgTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_lobpcg_correctness(self):
        # Random dimension for the tensor matrices (between 1x1 to 5x5 for simplicity)
        matrix_dim = random.randint(3, 5)
        
        # Random initial approximation size (columns for the eigenvectors)
        k_value = random.randint(1, matrix_dim// 3)
        n_value = k_value  # Set n_value to be same as k_value for our test
        
        # Generate random positive symmetric matrix A of size (matrix_dim, matrix_dim)
        A = torch.randn((matrix_dim, matrix_dim))
        A = (A + A.t()) / 2  # Making the matrix symmetric

        # 确保矩阵 A 是正定的（添加一个较大的对角元素）
        A += matrix_dim * torch.eye(matrix_dim)
        
        # Generate initial approximation eigenvector matrix X of size (matrix_dim, n_value)
        X = torch.randn((matrix_dim, n_value))

        # Run torch.lobpcg
        eigenvalues, eigenvectors = torch.lobpcg(A, X=X,)
        return eigenvalues, eigenvectors
    
    
    
    