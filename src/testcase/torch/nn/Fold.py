import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.Fold)
class TorchNnFoldTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_fold_correctness(self):
    # Randomly generate parameters for Fold
    output_height = random.randint(4, 10)
    output_width = random.randint(4, 10)
    kernel_height = random.randint(2, 4)
    kernel_width = random.randint(2, 4)
    stride_height = random.randint(1, 3)
    stride_width = random.randint(1, 3)
    padding_height = random.randint(0, 2)
    padding_width = random.randint(0, 2)
    dilation_height = random.randint(1, 2)
    dilation_width = random.randint(1, 2)

    # Calculate the number of blocks (L) based on the formula
    L_height = (output_height + 2 * padding_height - dilation_height * (kernel_height - 1) - 1) // stride_height + 1
    L_width = (output_width + 2 * padding_width - dilation_width * (kernel_width - 1) - 1) // stride_width + 1
    L = L_height * L_width

    # Randomly generate batch size and number of channels
    N = random.randint(1, 4)
    C = random.randint(1, 3)

    # Create random input tensor
    input_tensor = torch.randn(N, C * kernel_height * kernel_width, L)

    # Create Fold instance with generated parameters
    fold = torch.nn.Fold((output_height, output_width), (kernel_height, kernel_width), (dilation_height, dilation_width), (padding_height, padding_width), (stride_height, stride_width))

    # Apply Fold operation
    output_tensor = fold(input_tensor)
    return output_tensor
