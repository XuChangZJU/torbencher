import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.ZeroPad3d)
class TorchNnZeropad3dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_ZeroPad3d_correctness(self):
        # Random input size
        dim = 5
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Random padding as a 6-tuple
        padding_left = random.randint(1, 5)
        padding_right = random.randint(1, 5)
        padding_top = random.randint(1, 5)
        padding_bottom = random.randint(1, 5)
        padding_front = random.randint(1, 5)
        padding_back = random.randint(1, 5)
        padding = (padding_left, padding_right, padding_top, padding_bottom, padding_front, padding_back)

        # Create random input tensor
        input_tensor = torch.randn(input_size)

        # Create ZeroPad3d module
        zero_pad_3d = torch.nn.ZeroPad3d(padding)

        # Apply padding to the input tensor
        output_tensor = zero_pad_3d(input_tensor)

        return output_tensor
