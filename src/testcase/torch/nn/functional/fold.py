import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.functional.fold)
class TorchNnFunctionalFoldTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_fold_correctness(self):
        # Randomly generate parameters for fold operation
        batch_size = random.randint(1, 4)  # Random batch size
        channels = random.randint(1, 4)  # Random number of channels
        output_height = random.randint(5, 10)  # Random output height
        output_width = random.randint(5, 10)  # Random output width
        kernel_size = (random.randint(1, 3), random.randint(1, 3))  # Random kernel size
        stride = (random.randint(1, 3), random.randint(1, 3))  # Random stride
        padding = (random.randint(0, 2), random.randint(0, 2))  # Random padding
        dilation = (random.randint(1, 2), random.randint(1, 2))  # Random dilation

        # Calculate the number of sliding blocks
        input_height = (output_height + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1
        input_width = (output_width + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1] + 1

        # Generate random input tensor
        input_tensor = torch.randn(batch_size, channels * kernel_size[0] * kernel_size[1], input_height * input_width)

        # Perform fold operation
        result = torch.nn.functional.fold(input_tensor, (output_height, output_width), kernel_size, dilation, padding,
                                          stride)
        return result
