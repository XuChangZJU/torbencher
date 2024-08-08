import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.AdaptiveAvgPool2d)
class TorchNnAdaptiveavgpool2dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_AdaptiveAvgPool2d_correctness(self):
        # Random input size
        batch_size = random.randint(1, 3)
        num_channels = random.randint(1, 3)
        input_height = random.randint(4, 8)
        input_width = random.randint(4, 8)
        input_size = [batch_size, num_channels, input_height, input_width]

        # Random output size
        output_height = random.randint(1, input_height)  # output height should be smaller than input height
        output_width = random.randint(1, input_width)  # output width should be smaller than input width
        output_size = (output_height, output_width)

        # Generate random input tensor
        input_tensor = torch.randn(input_size)

        # Create AdaptiveAvgPool2d module
        adaptive_avg_pool = torch.nn.AdaptiveAvgPool2d(output_size)

        # Apply adaptive average pooling
        output_tensor = adaptive_avg_pool(input_tensor)

        return output_tensor
