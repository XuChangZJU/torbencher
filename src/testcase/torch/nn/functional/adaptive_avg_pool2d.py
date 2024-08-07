import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.functional.adaptive_avg_pool2d)
class TorchNnFunctionalAdaptiveUavgUpool2dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_adaptive_avg_pool2d_correctness(self):
        # Randomly generate dimensions for the input tensor
        batch_size = random.randint(1, 4)  # Random batch size between 1 and 4
        channels = random.randint(1, 4)  # Random number of channels between 1 and 4
        height = random.randint(5, 10)  # Random height between 5 and 10
        width = random.randint(5, 10)  # Random width between 5 and 10

        # Create a random input tensor with the generated dimensions
        input_tensor = torch.randn(batch_size, channels, height, width)

        # Randomly generate the target output size
        output_height = random.randint(1, height)  # Output height between 1 and input height
        output_width = random.randint(1, width)  # Output width between 1 and input width
        output_size = (output_height, output_width)

        # Apply adaptive average pooling
        result = torch.nn.functional.adaptive_avg_pool2d(input_tensor, output_size)
        return result
