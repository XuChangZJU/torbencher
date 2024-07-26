import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.CircularPad1d)
class TorchNnCircularpad1dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_CircularPad1d_correctness(self):
        # Randomly generate input tensor size
        batch_size = random.randint(1, 3)
        channels = random.randint(1, 3)
        length = random.randint(1, 10)
        input_size = [batch_size, channels, length]
    
        # Generate random input tensor
        input_tensor = torch.randn(input_size)
    
        # Generate random padding size
        padding_left = random.randint(0, length // 2)  # Ensure padding is not larger than half the length
        padding_right = random.randint(0, length // 2)
        padding = (padding_left, padding_right)
    
        # Apply CircularPad1d
        circular_pad_1d = torch.nn.CircularPad1d(padding)
        result = circular_pad_1d(input_tensor)
        return result
    
    
    
    