import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.Module)
class TorchNnModuleTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_nn_module_correctness(self):
        class RandomModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # Randomly generate the number of input and output channels for Conv2d layers
                in_channels1 = random.randint(1, 10)
                out_channels1 = random.randint(1, 10)  # This will be used for both conv1's output and conv2's input
                kernel_size = random.randint(1, 5)

                self.conv1 = torch.nn.Conv2d(in_channels1, out_channels1, kernel_size)
                self.conv2 = torch.nn.Conv2d(out_channels1, out_channels1,
                                             kernel_size)  # Use out_channels1 here as well

            def forward(self, x):
                x = torch.nn.functional.relu(self.conv1(x))
                return torch.nn.functional.relu(self.conv2(x))

        # Randomly generate input tensor dimensions
        batch_size = random.randint(1, 4)
        in_channels = random.randint(1, 10)
        height = random.randint(10, 20)
        width = random.randint(10, 20)
        input_tensor = torch.randn(batch_size, in_channels, height, width)

        model = RandomModel()
        # Ensure the input tensor's channels match the first convolution layer's expected input channels
        in_channels = model.conv1.in_channels  # Set in_channels to match the first layer's input requirement
        input_tensor = torch.randn(batch_size, in_channels, height, width)
        result = model(input_tensor)
        return result
