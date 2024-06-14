import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.ReflectionPad1d)
class TorchNnReflectionpad1dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_reflection_pad1d_correctness(self):
        # Randomly choose the number of dimensions (either 2 or 3)
        num_dims = random.choice([2, 3])
        
        # Randomly choose the size of each dimension
        C = random.randint(1, 5)  # Number of channels
        W_in = random.randint(5, 10)  # Width of the input tensor
        
        if num_dims == 2:
            input_size = [C, W_in]
        else:
            N = random.randint(1, 5)  # Batch size
            input_size = [N, C, W_in]
        
        # Generate a random input tensor
        input_tensor = torch.randn(input_size)
        
        # Randomly choose padding values
        padding_left = random.randint(1, 3)
        padding_right = random.randint(1, 3)
        padding = (padding_left, padding_right)
        
        # Create the ReflectionPad1d module
        reflection_pad = torch.nn.ReflectionPad1d(padding)
        
        # Apply the padding to the input tensor
        result = reflection_pad(input_tensor)
        
        return result
    