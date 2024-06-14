import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.bartlett_window)
class TorchBartlettwindowTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_bartlett_window_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1,5) # Random number of elements each dimension
        input_size=[num_of_elements_each_dim for i in range(dim)] 
    
        window_length = random.randint(1, 10)  # Random window_length value between 1 and 10
        periodic = random.choice([True, False])  # Random periodic value
    
        result = torch.bartlett_window(window_length, periodic)
        return result
    
    
    
    
    
    
    