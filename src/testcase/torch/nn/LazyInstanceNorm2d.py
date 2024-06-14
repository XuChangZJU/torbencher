import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.LazyInstanceNorm2d)
class TorchNnLazyinstancenorm2dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_lazy_instance_norm2d_correctness(self):
        # Randomly generate dimensions for the input tensor
        N = random.randint(1, 4)  # Batch size
        C = random.randint(1, 4)  # Number of channels
        H = random.randint(1, 5)  # Height
        W = random.randint(1, 5)  # Width
    
        # Generate a random input tensor with the specified dimensions
        input_tensor = torch.randn(N, C, H, W)
    
        # Create an instance of LazyInstanceNorm2d
        lazy_instance_norm2d = torch.nn.LazyInstanceNorm2d()
    
        # Apply the LazyInstanceNorm2d to the input tensor
        result = lazy_instance_norm2d(input_tensor)
        
        return result
    
    
    
    