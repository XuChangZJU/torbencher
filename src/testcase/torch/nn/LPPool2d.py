import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.LPPool2d)
class TorchNnLppool2dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_LPPool2d_correctness(self):
        # Random input size
        batch_size = random.randint(1, 10)
        channels = random.randint(1, 3)
        height = random.randint(10, 32)
        width = random.randint(10, 32)
        input_size = [batch_size, channels, height, width]
    
        # Random kernel size
        kernel_size = random.randint(1, min(height, width))
    
        # Random stride (ensure it's less than or equal to kernel_size)
        stride = random.randint(1, kernel_size)
    
        # Random norm
        norm = random.uniform(1.0, 2.0)
    
        # Create random input tensor
        input_tensor = torch.randn(input_size)
    
        # Create LPPool2d module
        lp_pool = torch.nn.LPPool2d(norm, kernel_size, stride=stride)
    
        # Perform LPPool2d operation
        result = lp_pool(input_tensor)
        return result
    