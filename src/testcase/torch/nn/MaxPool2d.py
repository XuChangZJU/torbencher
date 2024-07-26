import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.MaxPool2d)
class TorchNnMaxpool2dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_maxpool2d_correctness(self):
        # Randomly generate dimensions for the input tensor
        N = random.randint(1, 10)  # Batch size
        C = random.randint(1, 10)  # Number of channels
        H = random.randint(10, 20)  # Height of the input tensor
        W = random.randint(10, 20)  # Width of the input tensor

        # Randomly generate kernel size, stride, padding, and dilation within constraints
        kernel_size = (random.randint(2, min(H - 1, 5)), random.randint(2, min(W - 1, 5)))
        stride = (random.randint(1, min(kernel_size[0], 3)), random.randint(1, min(kernel_size[1], 3)))
        padding = (random.randint(0, min(kernel_size[0] // 2, (H - kernel_size[0]) // (stride[0] + 1))),
                   random.randint(0, min(kernel_size[1] // 2, (W - kernel_size[1]) // (stride[1] + 1))))
        dilation = (random.randint(1, 2), random.randint(1, 2))

        # Validate the configuration to ensure positive output dimensions
        H_out = ((H + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0]) + 1
        W_out = ((W + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1]) + 1
        assert H_out > 0 and W_out > 0, "Computed output dimensions are invalid"

        # Create a random input tensor
        input_tensor = torch.randn(N, C, H, W)

        # Create the MaxPool2d layer with the validated parameters
        maxpool_layer = torch.nn.MaxPool2d(kernel_size, stride, padding, dilation)

        # Apply the MaxPool2d layer to the input tensor
        output_tensor = maxpool_layer(input_tensor)

        return output_tensor
