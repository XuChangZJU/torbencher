import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.functional.upsample_bilinear)
class TorchNnFunctionalUpsamplebilinearTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_upsample_bilinear_correctness(self):
        # Randomly generate dimensions for the input tensor
        batch_size = random.randint(1, 4)
        channels = random.randint(1, 4)
        height = random.randint(1, 5)
        width = random.randint(1, 5)
        input_size = [batch_size, channels, height, width]

        # Generate a random input tensor
        input_tensor = torch.randn(input_size)

        # Randomly generate the output size
        output_height = random.randint(6, 10)
        output_width = random.randint(6, 10)
        output_size = (output_height, output_width)

        # Perform bilinear upsampling
        result = torch.nn.functional.upsample_bilinear(input_tensor, output_size)
        return result
