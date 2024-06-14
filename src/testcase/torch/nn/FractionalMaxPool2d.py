import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.FractionalMaxPool2d)
class TorchNnFractionalmaxpool2dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_FractionalMaxPool2d_correctness(self):
    # Random input size
    batch_size = random.randint(1, 3)
    channels = random.randint(1, 3)
    height = random.randint(10, 20)
    width = random.randint(10, 20)
    input_size = [batch_size, channels, height, width]

    # Random kernel size
    kernel_size = random.randint(1, min(height, width) // 2)

    # Random output size (make sure it's smaller than input size)
    output_height = random.randint(1, height - kernel_size + 1)
    output_width = random.randint(1, width - kernel_size + 1)
    output_size = (output_height, output_width)

    # Create FractionalMaxPool2d module
    fractional_max_pool = torch.nn.FractionalMaxPool2d(kernel_size, output_size)

    # Generate random input tensor
    input_tensor = torch.randn(input_size)

    # Apply FractionalMaxPool2d
    output_tensor = fractional_max_pool(input_tensor)

    return output_tensor
