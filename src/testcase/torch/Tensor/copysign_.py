import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.copysign_)
class TorchTensorCopysignTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_copysign__correctness(self):
        # Define the dimension and size of the input tensors randomly
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
    
        # Create random input tensors 
        input_tensor = torch.randn(input_size)
        other_tensor = torch.randn(input_size)  
    
        # Apply the copysign_ operation in-place
        input_tensor.copysign_(other_tensor)
    
        # Return the modified input_tensor to observe the effect of copysign_
        return input_tensor
    
    
    
    