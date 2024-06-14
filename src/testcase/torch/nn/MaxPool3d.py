import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.MaxPool3d)
class TorchNnMaxpool3dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_MaxPool3d_correctness(self):
        # Random input size
        dim = random.randint(3, 5)  
        num_of_elements_each_dim = random.randint(4, 8) 
        input_size = [num_of_elements_each_dim for i in range(dim)] 
    
        # Random kernel size
        kernel_size_d = random.randint(1, input_size[2] // 2)  # Ensuring kernel size is smaller than input dimensions
        kernel_size_h = random.randint(1, input_size[3] // 2)
        kernel_size_w = random.randint(1, input_size[4] // 2)
        kernel_size = (kernel_size_d, kernel_size_h, kernel_size_w)
    
        # Random stride
        stride_d = random.randint(1, kernel_size_d)  # Ensuring stride is less than or equal to kernel size
        stride_h = random.randint(1, kernel_size_h)
        stride_w = random.randint(1, kernel_size_w)
        stride = (stride_d, stride_h, stride_w)
    
        # Random padding
        padding_d = random.randint(0, kernel_size_d - 1)  # Ensuring padding is less than kernel size
        padding_h = random.randint(0, kernel_size_h - 1)
        padding_w = random.randint(0, kernel_size_w - 1)
        padding = (padding_d, padding_h, padding_w)
    
        # Random dilation
        dilation_d = random.randint(1, 2)  # Keeping dilation small for simplicity
        dilation_h = random.randint(1, 2)
        dilation_w = random.randint(1, 2)
        dilation = (dilation_d, dilation_h, dilation_w)
    
        # Randomly choose whether to return indices
        return_indices = random.choice([True, False])
    
        # Randomly choose ceil_mode
        ceil_mode = random.choice([True, False])
    
        input_tensor = torch.randn(input_size)
        max_pool_3d = torch.nn.MaxPool3d(kernel_size, stride, padding, dilation, return_indices, ceil_mode)
        result = max_pool_3d(input_tensor)
        return result 
    
    