import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.functional.lp_pool3d)
class TorchNnFunctionalLppool3dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_lp_pool3d_correctness(self):
        # Randomly generate dimensions for the input tensor
        batch_size = random.randint(1, 4)  # Random batch size
        channels = random.randint(1, 4)  # Random number of channels
        depth = random.randint(5, 10)  # Random depth of the 3D tensor
        height = random.randint(5, 10)  # Random height of the 3D tensor
        width = random.randint(5, 10)  # Random width of the 3D tensor

        # Generate random input tensor with the specified dimensions
        input_tensor = torch.randn(batch_size, channels, depth, height, width)

        # Randomly generate the power parameter p
        p = random.uniform(1.0, 3.0)  # Random p value between 1.0 and 3.0

        # Randomly generate kernel size, stride, and padding
        kernel_size = random.randint(2, 4)  # Random kernel size
        stride = random.randint(1, 3)  # Random stride
        padding = random.randint(0, 2)  # Random padding

        # Apply lp_pool3d with the generated parameters
        result = torch.nn.functional.lp_pool3d(input_tensor, norm_type=p, kernel_size=kernel_size, stride=stride,
                                               padding=padding)
        return result
