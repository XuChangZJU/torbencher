import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.hypot)
class TorchHypotTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_hypot_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1,5) # Random number of elements each dimension
        input_size=[num_of_elements_each_dim for i in range(dim)] 
    
        input_tensor = torch.randn(input_size)
        # Generate other tensor with broadcastable shape
        other_size = random.choice([input_size[:i] + [1] * (len(input_size) - i) for i in range(len(input_size) + 1)])  
        other_tensor = torch.randn(other_size)
        result = torch.hypot(input_tensor, other_tensor)
        return result
    
    
    
    
    
    
    