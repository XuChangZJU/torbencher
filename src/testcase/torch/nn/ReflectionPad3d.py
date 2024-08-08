import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.ReflectionPad3d)
class TorchNnReflectionpad3dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_reflection_pad3d_correctness(self):
        # Randomly generate dimensions for the input tensor
        N = random.randint(1, 4)  # Batch size
        C = random.randint(1, 4)  # Number of channels
        D_in = random.randint(3, 6)  # Depth of the input tensor
        H_in = random.randint(3, 6)  # Height of the input tensor
        W_in = random.randint(3, 6)  # Width of the input tensor

        # Ensure padding does not exceed half the size of the respective dimension to avoid errors
        # Adjust padding generation to ensure it's valid for the input dimensions
        max_pad = 1  # Start with a minimum valid padding to avoid division issues for very small dimensions
        if D_in > 8: max_pad = D_in // 2  # Example logic to allow larger padding for larger depths
        if H_in > 8: max_pad = H_in // 2  # Adjust for height
        if W_in > 8: max_pad = W_in // 2  # Adjust for width

        padding_left = random.randint(1, max_pad)
        padding_right = random.randint(1, max_pad)
        padding_top = random.randint(1, max_pad)
        padding_bottom = random.randint(1, max_pad)
        padding_front = random.randint(1, max_pad)
        padding_back = random.randint(1, max_pad)

        # Ensure the total padding along each axis doesn't exceed the dimension size
        padding_left, padding_right = sorted([padding_left, padding_right])[:min(2 * padding_left, D_in)]
        padding_top, padding_bottom = sorted([padding_top, padding_bottom])[:min(2 * padding_top, H_in)]
        padding_front, padding_back = sorted([padding_front, padding_back])[:min(2 * padding_front, W_in)]

        # Combine into a tuple for ReflectionPad3d
        padding = (padding_left, padding_right, padding_top, padding_bottom, padding_front, padding_back)

        # Create the input tensor with random values
        input_tensor = torch.randn(N, C, D_in, H_in, W_in)

        # Create the ReflectionPad3d module with the generated padding values
        reflection_pad3d = torch.nn.ReflectionPad3d(padding)

        # Apply the padding to the input tensor
        result = reflection_pad3d(input_tensor)

        # Ideally, you would also want to add some assertions here to verify the correctness of the padding operation.
        # For example, checking the output shape against expected dimensions after padding.

        return result
