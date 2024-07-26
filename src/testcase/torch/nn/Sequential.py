import torch
import random
from collections import OrderedDict
from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.Sequential)
class TorchNnSequentialTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_sequential_correctness(self):
        # Randomly generate dimensions for input tensor, ensuring channels match in Conv2d layer
        batch_size = random.randint(1, 4)
        channels = random.randint(1, 3)  # Ensure this matches the Conv2d input channel configuration
        height = random.randint(5, 10)
        width = random.randint(5, 10)

        # Create a random input tensor with the generated dimensions
        input_tensor = torch.randn(batch_size, channels, height, width)

        # Define a Sequential model with Conv2d layers, ensuring the first Conv2d layer's in_channels matches the input's channels
        in_channels_first_layer = channels
        out_channels_first_layer = random.randint(1, 10)
        kernel_size_first = random.randint(1, 3)

        in_channels_second_layer = out_channels_first_layer
        out_channels_second_layer = random.randint(1, 20)
        kernel_size_second = random.randint(1, 3)

        model = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels_first_layer, out_channels_first_layer, kernel_size=kernel_size_first),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels_second_layer, out_channels_second_layer, kernel_size=kernel_size_second),
            torch.nn.ReLU()
        )

        # Pass the input tensor through the Sequential model
        output_tensor = model(input_tensor)
        return output_tensor
