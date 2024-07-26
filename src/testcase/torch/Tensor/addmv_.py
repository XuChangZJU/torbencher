import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.addmv_)
class TorchTensorAddmvTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_addmv__correctness(self):
        # Randomly generate dimensions for the matrix and vector
        rows = random.randint(1, 5)  # Random number of rows for the matrix
        cols = random.randint(1, 5)  # Random number of columns for the matrix
    
        # Generate random tensors for the matrix and vector
        mat = torch.randn(rows, cols)
        vec = torch.randn(cols)
        
        # Generate a random tensor for the result
        result = torch.randn(rows)
        
        # Perform the in-place addmv_ operation
        result.addmv_(mat, vec)
        
        return result
    
    
    
    