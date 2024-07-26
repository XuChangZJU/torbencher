import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.functional.fractional_max_pool2d)
class TorchNnFunctionalFractionalmaxpool2dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_fractional_max_pool2d_correctness(self):
        # Random input size
        batch_size = random.randint(1, 10)
        num_channels = random.randint(1, 10)
        input_height = random.randint(10, 32)
        input_width = random.randint(10, 32)
        input_size = [batch_size, num_channels, input_height, input_width]

        # Random kernel size
        kernel_size = random.randint(2, 4)  # Make sure kernel size is smaller than input size

        # Random output size (make sure it's smaller than input size)
        output_height = random.randint(1, input_height // 2)
        output_width = random.randint(1, input_width // 2)
        output_size = (output_height, output_width)

        input_tensor = torch.randn(input_size)
        output_tensor, output_indices = torch.nn.functional.fractional_max_pool2d(input_tensor, kernel_size,
                                                                                  output_size=output_size,
                                                                                  return_indices=True)
        return output_tensor
