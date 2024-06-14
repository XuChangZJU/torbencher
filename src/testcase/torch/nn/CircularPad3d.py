import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.CircularPad3d)
class TorchNnCircularpad3dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_circular_pad3d_correctness(self):
        # Randomly generate dimensions for the input tensor
        N = random.randint(1, 4)  # Batch size
        C = random.randint(1, 4)  # Number of channels
        D_in = random.randint(4, 8)  # Depth
        H_in = random.randint(4, 8)  # Height
        W_in = random.randint(4, 8)  # Width
    
        # Randomly generate padding values
        padding_left = random.randint(1, 3)
        padding_right = random.randint(1, 3)
        padding_top = random.randint(1, 3)
        padding_bottom = random.randint(1, 3)
        padding_front = random.randint(1, 3)
        padding_back = random.randint(1, 3)
    
        # Create a random input tensor
        input_tensor = torch.randn(N, C, D_in, H_in, W_in)
    
        # Apply CircularPad3d with the generated padding values
        circular_pad3d = torch.nn.CircularPad3d((padding_left, padding_right, padding_top, padding_bottom, padding_front, padding_back))
        output_tensor = circular_pad3d(input_tensor)
    
        return output_tensor
    