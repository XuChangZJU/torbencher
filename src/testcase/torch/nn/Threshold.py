import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.Threshold)
class TorchNnThresholdTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_Threshold_correctness(self):
        # Define the dimension and size of the input tensor
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
    
        # Generate random input tensor
        input_tensor = torch.randn(input_size)
    
        # Generate random threshold and value
        threshold = random.uniform(-1, 1)  # Random threshold between -1 and 1
        value = random.uniform(-10, 10)   # Random value between -10 and 10
    
        # Create a Threshold module
        m = torch.nn.Threshold(threshold, value)
    
        # Apply the Threshold module to the input tensor
        output = m(input_tensor)
    
        return output
    
    
    
    