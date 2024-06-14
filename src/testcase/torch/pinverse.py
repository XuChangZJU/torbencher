import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.pinverse)
class TorchPinverseTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_pinverse_correctness(self):
        # Generate random dimension for the tensor
        dim = random.randint(1, 4)
        # Generate random number of rows and columns, ensuring the matrix is not singular
        m = random.randint(1, 5)
        n = random.randint(1, 5)
        input_size = [m, n]
    
        # Generate a random tensor of shape (m, n)
        input_tensor = torch.randn(input_size)
        
        # Calculate the pseudoinverse of the input tensor
        result = torch.pinverse(input_tensor)
        
        return result
    