import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.avg_pool1d)
class TorchNnFunctionalAvgpool1dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_avg_pool1d_correctness(self):
        # Randomly generate minibatch size and number of input channels
        minibatch_size = random.randint(1, 4)
        in_channels = random.randint(1, 4)
        
        # Randomly generate the input width
        input_width = random.randint(5, 10)
        
        # Randomly generate kernel size and ensure it's valid
        kernel_size = random.randint(1, input_width)
        
        # Randomly generate stride and ensure it's valid
        stride = random.randint(1, kernel_size)
        
        # Randomly generate padding and ensure it's valid
        padding = random.randint(0, kernel_size - 1)
        
        # Generate random input tensor
        input_tensor = torch.randn(minibatch_size, in_channels, input_width)
        
        # Apply avg_pool1d
        result = torch.nn.functional.avg_pool1d(input_tensor, kernel_size, stride, padding)
        
        return result
    
    
    
    