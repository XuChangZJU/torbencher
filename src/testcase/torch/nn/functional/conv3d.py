import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.functional.conv3d)
class TorchNnFunctionalConv3dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_conv3d_correctness(self):
        # Random input size
        minibatch = random.randint(1, 4)
        in_channels = random.randint(1, 4)
        iT = random.randint(10, 20)
        iH = random.randint(10, 20)
        iW = random.randint(10, 20)
        input_size = [minibatch, in_channels, iT, iH, iW]

        # Random filter size
        out_channels = random.randint(1, 4)
        kT = random.randint(1, 4)
        kH = random.randint(1, 4)
        kW = random.randint(1, 4)
        filter_size = [out_channels, in_channels, kT, kH, kW]

        input_tensor = torch.randn(input_size)
        filter_tensor = torch.randn(filter_size)
        result = torch.nn.functional.conv3d(input_tensor, filter_tensor)
        return result
