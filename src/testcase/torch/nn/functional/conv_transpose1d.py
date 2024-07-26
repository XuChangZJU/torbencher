import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.conv_transpose1d)
class TorchNnFunctionalConvtranspose1dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_conv_transpose1d_correctness(self):
        # Random input size
        minibatch = random.randint(1, 10)
        in_channels = random.randint(1, 10)
        iW = random.randint(1, 10)
        input_size = [minibatch, in_channels, iW]
    
        # Random weight size, out_channels should be divisible by groups
        out_channels = random.randint(1, 10) * random.randint(1, 10)
        kW = random.randint(1, iW)
        weight_size = [in_channels, out_channels // random.randint(1, out_channels), kW]
    
        input = torch.randn(input_size)
        weight = torch.randn(weight_size)
        result = torch.nn.functional.conv_transpose1d(input, weight)
        return result
    
    
    
    