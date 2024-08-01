import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.max_pool2d)
class TorchNnFunctionalMaxpool2dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_max_pool2d_correctness(self):
        # Randomly generate dimensions for the input tensor
        batch_size = random.randint(1, 4)  # Random batch size
        in_channels = random.randint(1, 4)  # Random number of input channels
        height = random.randint(5, 10)  # Random height of the input tensor
        width = random.randint(5, 10)  # Random width of the input tensor

        # Randomly generate parameters for max_pool2d with dynamic adjustment
        kernel_height = random.randint(2, 4)  # Random kernel height
        kernel_width = random.randint(2, 4)  # Random kernel width
        kernel_size = (kernel_height, kernel_width)
        stride_height = random.randint(1, kernel_height)  # Random stride height, must be <= kernel_height
        stride_width = random.randint(1, kernel_width)  # Random stride width, must be <= kernel_width
        padding_height = random.randint(0, kernel_height // 2)  # Random padding height, must be >= 0 and <= kernel_height / 2
        padding_width = random.randint(0, kernel_width // 2)  # Random padding width, must be >= 0 and <= kernel_width / 2
        dilation = 1  # For simplicity, we set dilation to 1 as larger values may cause issues

        # Dynamic adjustment to ensure valid output size
        while True:
            # Compute output size based on formula: output_size = floor((W + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)
            out_height = ((height + 2 * padding_height - dilation * (kernel_height - 1) - 1) // stride_height) + 1
            out_width = ((width + 2 * padding_width - dilation * (kernel_width - 1) - 1) // stride_width) + 1
            if out_height > 0 and out_width > 0:
                break
            # If output size is not valid, adjust stride or padding
            stride_height = random.randint(1, kernel_height)
            stride_width = random.randint(1, kernel_width)
            padding_height = random.randint(0, kernel_height // 2)
            padding_width = random.randint(0, kernel_width // 2)

        # Generate random input tensor
        input_tensor = torch.randn(batch_size, in_channels, height, width)

        # Apply max_pool2d
        result = torch.nn.functional.max_pool2d(input_tensor, kernel_size, (stride_height, stride_width), (padding_height, padding_width), dilation)

        return result