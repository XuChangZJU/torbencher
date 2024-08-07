import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.functional.adaptive_max_pool3d)
class TorchNnFunctionalAdaptiveUmaxUpool3dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_adaptive_max_pool3d_correctness(self):
        # Randomly generate dimensions for the input tensor
        depth = random.randint(4, 10)
        height = random.randint(4, 10)
        width = random.randint(4, 10)
        channels = random.randint(1, 3)
        batch_size = random.randint(1, 3)

        # Create a random input tensor with the generated dimensions
        input_tensor = torch.randn(batch_size, channels, depth, height, width)

        # Randomly generate the target output size
        output_depth = random.randint(1, depth)
        output_height = random.randint(1, height)
        output_width = random.randint(1, width)
        output_size = (output_depth, output_height, output_width)

        # Apply adaptive max pooling
        result = torch.nn.functional.adaptive_max_pool3d(input_tensor, output_size)

        return result
