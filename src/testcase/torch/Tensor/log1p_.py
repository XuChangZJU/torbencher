import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.log1p_)
class TorchTensorLog1pTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_log1p__correctness(self):
        # Generate random dimension and size for the tensor
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1,5)
        input_size=[num_of_elements_each_dim for i in range(dim)] 
    
        # Generate a random tensor 
        input_tensor = torch.randn(input_size)
        
        # Perform the operation
        input_tensor.log1p_()
        
        # Return the tensor after the in-place operation
        return input_tensor
    
    
    
    