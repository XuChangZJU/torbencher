import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.functional.avg_pool3d)
class TorchNnFunctionalAvgUpool3dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_avg_pool3d_correctness(self):
        # Random dimensions for the input tensor
        batch_size = random.randint(1, 4)
        in_channels = random.randint(1, 4)
        depth = random.randint(5, 10)
        height = random.randint(5, 10)
        width = random.randint(5, 10)

        # Random kernel size
        kernel_size = (random.randint(2, 4), random.randint(2, 4), random.randint(2, 4))

        # Random stride, default to kernel size if not specified
        stride = (random.randint(1, 3), random.randint(1, 3), random.randint(1, 3))

        # Random padding, ensuring it is at most half of the kernel size
        padding = (random.randint(0, kernel_size[0] // 2),
                   random.randint(0, kernel_size[1] // 2),
                   random.randint(0, kernel_size[2] // 2))

        # Generate random input tensor
        input_tensor = torch.randn(batch_size, in_channels, depth, height, width)

        # Apply avg_pool3d
        result = torch.nn.functional.avg_pool3d(input_tensor, kernel_size, stride, padding)

        return result
