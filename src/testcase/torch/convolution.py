
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.convolution)
class TorchConvolutionTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_convolution_correctness(self):
        dim = random.randint(1, 10)
        input = torch.randn(dim)
        weight = torch.randn(dim)
        bias = torch.randn(dim)
        stride = random.randint(1, 10)
        padding = random.randint(0, 10)
        dilation = random.randint(1, 10)
        groups = random.randint(1, 10)
        result = torch.convolution(input, weight, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_convolution_large_scale(self):
        dim = random.randint(100, 1000)
        input = torch.randn(dim)
        weight = torch.randn(dim)
        bias = torch.randn(dim)
        stride = random.randint(1, 10)
        padding = random.randint(0, 10)
        dilation = random.randint(1, 10)
        groups = random.randint(1, 10)
        result = torch.convolution(input, weight, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
        return result

