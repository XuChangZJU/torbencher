import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.Fold)
class TorchNnFoldTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_fold_correctness(self):
        while True:
            # Randomly generate parameters ensuring they are within a valid range to likely avoid L <= 0
            output_height = random.randint(1, 16)
            output_width = random.randint(1, 16)
            kernel_height = random.randint(1, min(7, output_height))  # Ensure kernel doesn't exceed output dimension
            kernel_width = random.randint(1, min(7, output_width))
            stride_height = random.randint(1, kernel_height)
            stride_width = random.randint(1, kernel_width)
            padding_height = random.randint(0, kernel_height - 1)
            padding_width = random.randint(0, kernel_width - 1)
            dilation_height = 1  # For simplicity, keeping dilation as 1
            dilation_width = 1
            
            # Calculate L based on generated parameters
            L_height = (output_height + 2 * padding_height - dilation_height * (kernel_height - 1) - 1) // stride_height + 1
            L_width = (output_width + 2 * padding_width - dilation_width * (kernel_width - 1) - 1) // stride_width + 1
            L = L_height * L_width

            # If L is valid, break the loop and proceed; otherwise, try again
            if L > 0:
                break

        # Randomly generate batch size and number of channels
        N = random.randint(1, 4)
        C = random.randint(1, 3)
        
        # Create random input tensor based on calculated L
        input_tensor = torch.randn(N, C * kernel_height * kernel_width, L)
        
        # Create Fold instance with the validated parameters
        fold = torch.nn.Fold(output_size=(output_height, output_width), 
                            kernel_size=(kernel_height, kernel_width), 
                            dilation=(dilation_height, dilation_width), 
                            padding=(padding_height, padding_width), 
                            stride=(stride_height, stride_width))
        
        # Apply Fold operation
        output_tensor = fold(input_tensor)
        return output_tensor
    
    
    
    