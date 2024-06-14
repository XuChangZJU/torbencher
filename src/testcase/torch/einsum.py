import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.einsum)
class TorchEinsumTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_einsum_correctness_1(self):
    # Random dimensions for the tensors
    dim1 = random.randint(1, 3)  
    dim2 = random.randint(1, 3)  
    
    tensor1 = torch.randn(dim1, dim2)
    tensor2 = torch.randn(dim2, dim1)
    
    # Equation for matrix multiplication
    equation = 'ij,jk->ik'
    result = torch.einsum(equation, tensor1, tensor2)
    return result
