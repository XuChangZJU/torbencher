import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.LazyInstanceNorm3d)
class TorchNnLazyinstancenorm3dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_lazy_instance_norm_3d_correctness(self):
        # Randomly generate dimensions for the input tensor
        N = random.randint(1, 4)  # Batch size
        C = random.randint(1, 4)  # Number of channels
        D = random.randint(1, 5)  # Depth
        H = random.randint(1, 5)  # Height
        W = random.randint(1, 5)  # Width
    
        # Create a random input tensor with the generated dimensions
        input_tensor = torch.randn(N, C, D, H, W)
    
        # Initialize LazyInstanceNorm3d with default parameters
        lazy_instance_norm = torch.nn.LazyInstanceNorm3d()
    
        # Apply the LazyInstanceNorm3d to the input tensor
        result = lazy_instance_norm(input_tensor)
        
        return result
    