import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.AvgPool3d)
class TorchNnAvgpool3dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_avgpool3d_correctness(self):
        # Randomly generate dimensions for the input tensor
        N = random.randint(1, 5)  # Batch size
        C = random.randint(1, 5)  # Number of channels
        D_in = random.randint(10, 20)  # Depth of the input tensor
        H_in = random.randint(10, 20)  # Height of the input tensor
        W_in = random.randint(10, 20)  # Width of the input tensor

        # Randomly generate kernel size, stride, and padding
        kernel_size = (random.randint(4, 5), random.randint(4, 5), random.randint(4, 5))
        stride = (random.randint(1, 3), random.randint(1, 3), random.randint(1, 3))
        padding = (random.randint(0, 2), random.randint(0, 2), random.randint(0, 2))

        # Create the AvgPool3d layer
        avg_pool3d = torch.nn.AvgPool3d(kernel_size, stride, padding)

        # Generate a random input tensor
        input_tensor = torch.randn(N, C, D_in, H_in, W_in)

        # Apply the AvgPool3d layer to the input tensor
        output_tensor = avg_pool3d(input_tensor)

        return output_tensor
