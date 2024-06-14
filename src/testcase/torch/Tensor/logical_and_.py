import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.logical_and_)
class TorchTensorLogicalandTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_logical_and__correctness(self):
        # Randomly generate input tensor size
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
    
        # Generate random tensors
        input_tensor = torch.randint(0, 2, input_size, dtype=torch.bool) # Generate random tensor with element 0 or 1
        other_tensor = torch.randint(0, 2, input_size, dtype=torch.bool) # Generate random tensor with element 0 or 1
    
        # Perform logical_and_ operation
        input_tensor.logical_and_(other_tensor)
        
        return input_tensor
    
    
    
    