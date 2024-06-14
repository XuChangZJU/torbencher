import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.retains_grad)
class TorchTensorRetainsgradTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_retains_grad_correctness(self):
        # Randomly generate tensor dimension and size
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
    
        # Create a tensor
        tensor = torch.randn(input_size)
        
        # Randomly set requires_grad to True or False
        tensor.requires_grad = random.choice([True, False])
        
        # Perform an operation that creates a non-leaf tensor
        result = tensor + 2 
        
        # Check if retains_grad is as expected
        return result.retains_grad
    
    
    
    