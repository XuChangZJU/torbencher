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
    
    def test_einsum_correctness_2(self):
        dim1 = random.randint(1, 3)  # Random dimension
        dim2 = random.randint(1, 3)
        tensor1 = torch.randn(dim1, dim1)
        
        # Equation for diagonal summation
        equation = 'ii'
        result = torch.einsum(equation, tensor1)
        return result
    
    def test_einsum_correctness_3(self):
        dim1 = random.randint(1, 3) 
        dim2 = random.randint(1, 3)
        
        tensor1 = torch.randn(dim1)
        tensor2 = torch.randn(dim2)
        
        # Equation for outer product
        equation = 'i,j->ij'
        result = torch.einsum(equation, tensor1, tensor2)
        return result
    
    def test_einsum_correctness_4(self):
        batch_size = random.randint(1, 3)
        dim1 = random.randint(1, 3)
        dim2 = random.randint(1, 3)
        dim3 = random.randint(1, 3)
        
        tensor1 = torch.randn(batch_size, dim1, dim2)
        tensor2 = torch.randn(batch_size, dim2, dim3)
        
        # Equation for batch matrix multiplication
        equation = 'bij,bjk->bik'
        result = torch.einsum(equation, tensor1, tensor2)
        return result
    
    def test_einsum_correctness_5(self):
        batch_size = random.randint(1, 3)
        dim1 = random.randint(1, 3)
        dim2 = random.randint(1, 3)
        
        tensor = torch.randn(batch_size, dim1, dim2)
        
        # Equation for batch permute (transpose last two dims)
        equation = '...ij->...ji'
        result = torch.einsum(equation, tensor)
        return result
    
    # Calling test cases
    
    
    
    
    
    
    