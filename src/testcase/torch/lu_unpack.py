import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.lu_unpack)
class TorchLuunpackTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_lu_unpack_correctness(self):
        # Randomly choose the dimensions for the tensor
        batch_size = random.randint(1, 4)  # Batch size between 1 and 4
        m = random.randint(2, 4)  # Rows of the matrix between 2 and 4
        n = random.randint(2, 4)  # Columns of the matrix between 2 and 4
        
        # Randomly generate a batch of m x n matrices
        A = torch.randn((batch_size, m, n))
        
        # Calculate the LU factorization and pivots
        LU_data, LU_pivots = torch.linalg.lu_factor(A)
        
        # Unpack the LU factorization
        P, L, U = torch.lu_unpack(LU_data, LU_pivots)
        
        return {'original_matrix': A, 'P_matrix': P, 'L_matrix': L, 'U_matrix': U}
    
    
    
    