import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.functional.adaptive_avg_pool3d)
class TorchNnFunctionalAdaptiveUavgUpool3dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_adaptive_avg_pool3d_correctness(self):
        # Randomly generate dimensions for the input tensor
        batch_size = random.randint(1, 4)  # Random batch size
        channels = random.randint(1, 4)  # Random number of channels
        depth = random.randint(4, 8)  # Random depth of the 3D tensor
        height = random.randint(4, 8)  # Random height of the 3D tensor
        width = random.randint(4, 8)  # Random width of the 3D tensor

        # Create a random input tensor with the generated dimensions
        input_tensor = torch.randn(batch_size, channels, depth, height, width)

        # Randomly generate the target output size
        output_depth = random.randint(1, depth)
        output_height = random.randint(1, height)
        output_width = random.randint(1, width)
        output_size = (output_depth, output_height, output_width)

        # Apply adaptive average pooling
        result = torch.nn.functional.adaptive_avg_pool3d(input_tensor, output_size)
        return result
