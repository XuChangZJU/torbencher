import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.convtranspose3d)
class TorchNnFunctionalConvtranspose3dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_conv_transpose3d_correctness(self):
    # Random input size
    minibatch = random.randint(1, 3)
    in_channels = random.randint(1, 5)
    iT = random.randint(1, 10)
    iH = random.randint(1, 10)
    iW = random.randint(1, 10)
    input_size = [minibatch, in_channels, iT, iH, iW]

    # Random weight size, output_channels is divisible by groups
    kT = random.randint(1, 5)
    kH = random.randint(1, 5)
    kW = random.randint(1, 5)
    groups = random.randint(1, in_channels)
    out_channels = random.randint(1, 5) * groups
    weight_size = [in_channels, out_channels // groups, kT, kH, kW]

    input_tensor = torch.randn(input_size)
    weight_tensor = torch.randn(weight_size)
    result = torch.nn.functional.conv_transpose3d(input_tensor, weight_tensor)
    return result
