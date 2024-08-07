import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.max_pool3d)
class TorchNnFunctionalMaxUpool3dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_max_pool3d_correctness(self):
        # Random dimensions for the input tensor
        batch_size = random.randint(1, 4)
        in_channels = random.randint(1, 4)
        depth = random.randint(5, 10)
        height = random.randint(5, 10)
        width = random.randint(5, 10)
        input_size = [batch_size, in_channels, depth, height, width]

        # Random kernel size, ensuring it's smaller than the input dimensions
        kernel_size = (
            random.randint(2, min(4, depth)),
            random.randint(2, min(4, height)),
            random.randint(2, min(4, width))
        )

        # Random stride, ensuring it's smaller than or equal to the kernel size
        stride = (
            random.randint(1, kernel_size[0]),
            random.randint(1, kernel_size[1]),
            random.randint(1, kernel_size[2])
        )

        # Random padding, ensuring it's valid
        padding = (
            random.randint(0, kernel_size[0] // 2),
            random.randint(0, kernel_size[1] // 2),
            random.randint(0, kernel_size[2] // 2)
        )

        # Dynamic adjustment to ensure valid output size
        while True:
            # Compute output size based on formula: output_size = floor((W + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)
            out_depth = ((depth + 2 * padding[0] - 1 * (kernel_size[0] - 1) - 1) // stride[0]) + 1
            out_height = ((height + 2 * padding[1] - 1 * (kernel_size[1] - 1) - 1) // stride[1]) + 1
            out_width = ((width + 2 * padding[2] - 1 * (kernel_size[2] - 1) - 1) // stride[2]) + 1
            if out_depth > 0 and out_height > 0 and out_width > 0:
                break
            # If output size is not valid, adjust stride or padding
            stride = (
                random.randint(1, kernel_size[0]),
                random.randint(1, kernel_size[1]),
                random.randint(1, kernel_size[2])
            )
            padding = (
                random.randint(0, kernel_size[0] // 2),
                random.randint(0, kernel_size[1] // 2),
                random.randint(0, kernel_size[2] // 2)
            )

        # Generate random input tensor
        input_tensor = torch.randn(input_size)

        # Random dilation, ensuring it's greater than 0
        dilation = (1, 1, 1)  # Simplified for consistency and to avoid potential issues

        # Random ceil_mode and return_indices
        ceil_mode = random.choice([True, False])
        return_indices = random.choice([True, False])

        # Apply max_pool3d
        result = torch.nn.functional.max_pool3d(input_tensor, kernel_size, stride, padding, dilation, ceil_mode,
                                                return_indices)
        return result
