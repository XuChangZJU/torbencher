
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.MaxPool2d)
class TorchMaxPool2dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_maxpool2d_correctness(self):
        kernel_size = (random.randint(1, 10), random.randint(1, 10))
        stride = (random.randint(1, kernel_size[0]), random.randint(1, kernel_size[1]))
        padding = (random.randint(0, kernel_size[0]), random.randint(0, kernel_size[1]))
        dilation = (random.randint(1, kernel_size[0]), random.randint(1, kernel_size[1]))
        input_tensor = torch.randn(random.randint(1, 10), random.randint(1, 10), random.randint(1, 10), random.randint(1, 10))
        max_pool = torch.nn.MaxPool2d(kernel_size, stride=stride, padding=padding, dilation=dilation)
        result = max_pool(input_tensor)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_maxpool2d_large_scale(self):
        kernel_size = (random.randint(100, 1000), random.randint(100, 1000))
        stride = (random.randint(10, kernel_size[0]), random.randint(10, kernel_size[1]))
        padding = (random.randint(0, kernel_size[0]), random.randint(0, kernel_size[1]))
        dilation = (random.randint(10, kernel_size[0]), random.randint(10, kernel_size[1]))
        input_tensor = torch.randn(random.randint(1000, 10000), random.randint(100, 1000), random.randint(100, 1000), random.randint(100, 1000))
        max_pool = torch.nn.MaxPool2d(kernel_size, stride=stride, padding=padding, dilation=dilation)
        result = max_pool(input_tensor)
        return result

