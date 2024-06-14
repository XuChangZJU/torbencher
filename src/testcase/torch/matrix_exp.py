import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.matrix_exp)
class TorchMatrixexpTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_matrix_exp_correctness(self):
        # Generate a random square matrix
        dim = random.randint(1, 4)
        input_size = [dim, dim]  
        input_tensor = torch.randn(input_size)
        result = torch.matrix_exp(input_tensor)
        return result
    
    
    
    
    
    
    