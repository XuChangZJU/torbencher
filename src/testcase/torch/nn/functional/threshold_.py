import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.threshold_)
class TorchNnFunctionalThresholdTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_threshold_correctness(self):
        # Randomly generate input tensor dimension
        dim = random.randint(1, 4)
        # Randomly generate number of elements for each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Generate input size
        input_size = [num_of_elements_each_dim for i in range(dim)]
    
        # Generate random input tensor
        input_tensor = torch.randn(input_size)
        # Generate random threshold value
        threshold = random.uniform(-1.0, 1.0)  # Threshold value should be within the range of input tensor values
        # Generate random value to replace with
        value = random.uniform(-10.0, 10.0)  # Value can be any random number
    
        # Apply threshold_ operation
        result = torch.nn.functional.threshold_(input_tensor, threshold, value)
    
        return result
    
    
    
    