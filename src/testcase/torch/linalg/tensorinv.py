import torch
import random
import math

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.linalg.tensorinv)
class TorchLinalgTensorinvTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_tensorinv_correctness(self):
        # Randomly generate dimensions for the tensor
        ind = random.randint(1, 4)  # Random index for tensor inversion
    
        # Ensure the product of dimensions before and after 'ind' are equal
        size1 = [random.randint(1, 5) for _ in range(ind)]
        prod_size1 = math.prod(size1)
        
        # Generate size2 such that its product matches prod_size1
        size2 = []
        while math.prod(size2) != prod_size1:
            size2 = [random.randint(1, 5) for _ in range(random.randint(1, 4))]
        
        input_size = size1 + size2
    
        # Create a tensor with the generated size
        A = torch.randn(input_size)
        
        # Compute the tensor inverse
        Ainv = torch.linalg.tensorinv(A, ind)
        
        # Return the shape of the result to verify correctness
        return Ainv.shape
    