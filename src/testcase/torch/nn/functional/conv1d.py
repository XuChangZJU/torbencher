import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.conv1d)
class TorchNnFunctionalConv1dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_conv1d_correctness(self):
    # Random input size
    batch_size = random.randint(1, 10)
    in_channels = random.randint(1, 10)
    input_width = random.randint(1, 10)
    input_size = [batch_size, in_channels, input_width]

    # Random filter size
    out_channels = random.randint(1, 10)
    kernel_size = random.randint(1, input_width)  # kernel_size <= input_width
    filter_size = [out_channels, in_channels, kernel_size]

    input_tensor = torch.randn(input_size)
    filter_tensor = torch.randn(filter_size)
    result = torch.nn.functional.conv1d(input_tensor, filter_tensor)
    return result
