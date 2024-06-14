import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.maxunpool3d)
class TorchNnFunctionalMaxunpool3dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_max_unpool3d_correctness(self):
        # Randomly generate dimensions for the input tensor
        batch_size = random.randint(1, 4)
        channels = random.randint(1, 4)
        depth = random.randint(4, 8)
        height = random.randint(4, 8)
        width = random.randint(4, 8)
        
        # Generate random input tensor and indices tensor
        input_tensor = torch.randn(batch_size, channels, depth, height, width)
        indices_tensor = torch.randint(0, depth * height * width, (batch_size, channels, depth, height, width))
        
        # Define kernel size, stride, and padding for max pooling
        kernel_size = random.randint(2, 4)
        stride = kernel_size  # To ensure valid unpooling, stride should be equal to kernel size
        padding = 0  # No padding for simplicity
        
        # Perform max unpooling
        output_size = (depth, height, width)  # Output size should match the original input size before pooling
        result = torch.nn.functional.max_unpool3d(input_tensor, indices_tensor, kernel_size, stride, padding, output_size)
        
        return result
    