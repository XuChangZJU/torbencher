import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.functional.conv_transpose2d)
class TorchNnFunctionalConvUtranspose2dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_conv_transpose2d_correctness(self):
        # Random input size
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Generate random parameters for conv_transpose2d
        minibatch = random.randint(1, 10)
        in_channels = random.randint(1, 10)
        iH = random.randint(1, 10)
        iW = random.randint(1, 10)
        out_channels = random.randint(1, 10)  # out_channels should be divisible by groups
        kH = random.randint(1, iH)
        kW = random.randint(1, iW)

        # Create random input tensor
        input_tensor = torch.randn([minibatch, in_channels, iH, iW])

        # Create random weight tensor
        weight_tensor = torch.randn([in_channels, out_channels, kH, kW])

        result = torch.nn.functional.conv_transpose2d(input_tensor, weight_tensor)
        return result
