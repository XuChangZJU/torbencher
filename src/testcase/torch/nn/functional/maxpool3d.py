import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.maxpool3d)
class TorchNnFunctionalMaxpool3dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_max_pool3d_correctness(self):
        # Random dimensions for the input tensor
        batch_size = random.randint(1, 4)
        in_channels = random.randint(1, 4)
        depth = random.randint(5, 10)
        height = random.randint(5, 10)
        width = random.randint(5, 10)
        input_size = [batch_size, in_channels, depth, height, width]
    
        # Generate random input tensor
        input_tensor = torch.randn(input_size)
    
        # Random kernel size, ensuring it's smaller than the input dimensions
        kernel_size = (random.randint(2, 4), random.randint(2, 4), random.randint(2, 4))
    
        # Random stride, ensuring it's smaller than the kernel size
        stride = (random.randint(1, kernel_size[0]), random.randint(1, kernel_size[1]), random.randint(1, kernel_size[2]))
    
        # Random padding, ensuring it's valid
        padding = random.randint(0, min(kernel_size) // 2)
    
        # Random dilation, ensuring it's greater than 0
        dilation = random.randint(1, 2)
    
        # Random ceil_mode and return_indices
        ceil_mode = random.choice([True, False])
        return_indices = random.choice([True, False])
    
        # Apply max_pool3d
        result = torch.nn.functional.max_pool3d(input_tensor, kernel_size, stride, padding, dilation, ceil_mode, return_indices)
        return result
    