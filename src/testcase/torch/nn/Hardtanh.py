import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.Hardtanh)
class TorchNnHardtanhTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_Hardtanh_correctness(self):
        # Define the dimension and size of the input tensor
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
    
        # Generate random input tensor
        input_tensor = torch.randn(input_size)
    
        # Generate random min_val and max_val, ensuring min_val < max_val
        min_val = random.uniform(-10.0, 0.0)
        max_val = random.uniform(0.0, 10.0) 
    
        # Apply Hardtanh operation
        m = torch.nn.Hardtanh(min_val, max_val)
        output = m(input_tensor)
        
        return output
    