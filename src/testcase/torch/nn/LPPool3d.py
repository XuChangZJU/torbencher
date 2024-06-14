import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.LPPool3d)
class TorchNnLppool3dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_LPPool3d_correctness(self):
        # Randomly generate the power parameter p
        p = random.uniform(1.0, 10.0)
        
        # Randomly generate kernel size and stride
        kernel_size = (random.randint(2, 5), random.randint(2, 5), random.randint(2, 5))
        stride = (random.randint(1, 3), random.randint(1, 3), random.randint(1, 3))
        
        # Randomly generate input tensor dimensions
        N = random.randint(1, 4)  # Batch size
        C = random.randint(1, 4)  # Number of channels
        D_in = random.randint(10, 20)  # Depth
        H_in = random.randint(10, 20)  # Height
        W_in = random.randint(10, 20)  # Width
        
        # Create random input tensor
        input_tensor = torch.randn(N, C, D_in, H_in, W_in)
        
        # Create LPPool3d layer
        lp_pool = torch.nn.LPPool3d(p, kernel_size, stride)
        
        # Apply LPPool3d to input tensor
        output_tensor = lp_pool(input_tensor)
        
        return output_tensor
    