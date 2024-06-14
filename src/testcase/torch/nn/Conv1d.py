import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.Conv1d)
class TorchNnConv1dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_Conv1d_correctness(self):
    # Random input size (N, C_in, L_in)
    batch_size = random.randint(1, 10)
    in_channels = random.randint(1, 32)
    in_length = random.randint(1, 64)
    input_size = [batch_size, in_channels, in_length]

    # Random Conv1d parameters
    out_channels = random.randint(1, 32)  
    kernel_size = random.randint(1, in_length)  # kernel_size should be less than or equal to in_length
    stride = random.randint(1, in_length // kernel_size)  # Ensure stride is valid for kernel_size and in_length

    # Create Conv1d module and random input tensor
    conv1d = torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride)
    input_tensor = torch.randn(input_size)

    # Perform convolution
    output_tensor = conv1d(input_tensor)
    
    return output_tensor
