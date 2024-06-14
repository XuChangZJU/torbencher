import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.AdaptiveMaxPool3d)
class TorchNnAdaptivemaxpool3dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_adaptive_max_pool3d_correctness(self):
    # Randomly generate input dimensions
    N = random.randint(1, 4)  # Batch size
    C = random.randint(1, 10)  # Number of channels
    D_in = random.randint(5, 10)  # Depth of input
    H_in = random.randint(5, 10)  # Height of input
    W_in = random.randint(5, 10)  # Width of input

    # Randomly generate output dimensions
    D_out = random.randint(1, D_in)  # Depth of output
    H_out = random.randint(1, H_in)  # Height of output
    W_out = random.randint(1, W_in)  # Width of output

    # Create random input tensor
    input_tensor = torch.randn(N, C, D_in, H_in, W_in)

    # Create AdaptiveMaxPool3d layer with random output size
    adaptive_max_pool3d = torch.nn.AdaptiveMaxPool3d((D_out, H_out, W_out))

    # Apply the layer to the input tensor
    output_tensor = adaptive_max_pool3d(input_tensor)

    return output_tensor
