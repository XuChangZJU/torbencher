import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.addr_)
class TorchTensorAddrTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_addr_inplace_correctness(self):
        # Randomly generate dimensions for the matrix and vectors
        matrix_dim = random.randint(1, 4)
        vec_dim = random.randint(1, 4)
        
        # Randomly generate the size of each dimension
        matrix_size = [random.randint(1, 5) for _ in range(matrix_dim)]
        vec1_size = [random.randint(1, 5) for _ in range(vec_dim)]
        vec2_size = [random.randint(1, 5) for _ in range(vec_dim)]
        
        # Generate random tensors for the matrix and vectors
        matrix = torch.randn(matrix_size)
        vec1 = torch.randn(vec1_size)
        vec2 = torch.randn(vec2_size)
        
        # Perform the in-place addr operation
        result = matrix.addr_(vec1, vec2)
        return result
    
    
    
    