import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.isreal)
class TorchTensorIsrealTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_isreal_correctness(self):
        # Generate random dimension and size for the tensor
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
    
        # Create a real tensor
        real_tensor = torch.randn(input_size) 
        
        # Create a complex tensor
        complex_tensor = torch.randn(input_size) + torch.randn(input_size) * 1j 
    
        # Check if the tensors are real using isreal()
        result_real = real_tensor.isreal()
        result_complex = complex_tensor.isreal()
    
        return result_real, result_complex
    