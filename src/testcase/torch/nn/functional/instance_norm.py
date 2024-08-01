import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.functional.instance_norm)
class TorchNnFunctionalInstancenormTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_instance_norm_correctness(self):
        # Randomly choose the dimension (1, 2, or 3) for InstanceNorm
        dim = random.randint(1, 3)

        # Randomly generate the batch size and number of channels
        batch_size = random.randint(2, 4)
        num_channels = random.randint(2, 4)

        if dim == 1:
            # For InstanceNorm1d, generate random length for each sequence
            length = random.randint(2, 10)
            input_size = [batch_size, num_channels, length]
        elif dim == 2:
            # For InstanceNorm2d, generate random height and width for each image
            height = random.randint(2, 10)
            width = random.randint(2, 10)
            input_size = [batch_size, num_channels, height, width]
        else:
            # For InstanceNorm3d, generate random depth, height, and width for each volume
            depth = random.randint(2, 10)
            height = random.randint(2, 10)
            width = random.randint(2, 10)
            input_size = [batch_size, num_channels, depth, height, width]

        # Generate random input tensor with the specified size
        input_tensor = torch.randn(input_size)

        # Apply instance normalization
        result = torch.nn.functional.instance_norm(input_tensor)

        return result
