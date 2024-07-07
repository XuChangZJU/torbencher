import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.linalg.pinv)
class TorchLinalgPinvTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_torch_linalg_pinv_correctness(self):
        # Define the dimensions of the input matrix
        dim1 = random.randint(1, 10)  
        dim2 = random.randint(1, 10)
        # Generate a random matrix 
        input_tensor = torch.randn(dim1, dim2)
        # Calculate the pseudoinverse of the matrix
        result = torch.linalg.pinv(input_tensor)
        return result
    
    
    
    