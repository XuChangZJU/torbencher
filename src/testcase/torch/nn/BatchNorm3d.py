import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.BatchNorm3d)
class TorchNnBatchnorm3dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_BatchNorm3d_correctness(self):
        # Random input size (N, C, D, H, W)
        num_batches = random.randint(2, 10)
        num_channels = random.randint(2, 10)
        depth = random.randint(2, 10)
        height = random.randint(2, 10)
        width = random.randint(2, 10)
        input_size = (num_batches, num_channels, depth, height, width)
    
        # Random input tensor
        input_tensor = torch.randn(input_size)
    
        # Create a BatchNorm3d layer
        batch_norm_3d = torch.nn.BatchNorm3d(num_channels)
    
        # Apply batch normalization
        output_tensor = batch_norm_3d(input_tensor)
    
        return output_tensor
    
    
    
    