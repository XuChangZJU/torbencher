import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.LPPool1d)
class TorchNnLppool1dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_LPPool1d_correctness(self):
        # Random input size
        batch_size = random.randint(1, 10)
        channels = random.randint(1, 10)
        length = random.randint(10, 20)
        input_size = (batch_size, channels, length)
    
        # Random kernel_size and stride
        kernel_size = random.randint(1, length)
        stride = random.randint(1, kernel_size)  # Ensure stride <= kernel_size
    
        # Random p value
        p = random.uniform(1.0, 10.0)
    
        # Create input tensor
        input_tensor = torch.randn(input_size)
    
        # Create LPPool1d module
        lp_pool = torch.nn.LPPool1d(kernel_size, stride)
    
        # Perform LPPool1d operation
        output_tensor = lp_pool(input_tensor)
    
        return output_tensor
    
    
    
    