import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.ConstantPad1d)
class TorchNnConstantpad1dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_ConstantPad1d_correctness(self):
        # Randomly generate input tensor size
        batch_size = random.randint(1, 3) # Random batch size
        channels = random.randint(1, 3) # Random number of channels
        input_width = random.randint(1, 5) # Random input width
        input_size = [batch_size, channels, input_width]
    
        # Generate random input tensor
        input_tensor = torch.randn(input_size)
    
        # Generate random padding
        padding_left = random.randint(1, 3) # Random padding size for left side
        padding_right = random.randint(1, 3) # Random padding size for right side
        padding = (padding_left, padding_right)
    
        # Generate random padding value
        padding_value = random.uniform(0.1, 10.0) # Random padding value
    
        # Apply ConstantPad1d
        constant_pad_1d = torch.nn.ConstantPad1d(padding, padding_value)
        result = constant_pad_1d(input_tensor)
        return result
    
    
    
    