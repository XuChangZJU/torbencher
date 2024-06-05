
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.FractionalMaxPool2d)
class TorchFractionalMaxPool2dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_fractionalmaxpool2d_correctness(self):
        kernel_size = (random.randint(1, 10), random.randint(1, 10))
        output_size = (random.randint(1, 10), random.randint(1, 10))
        input_tensor = torch.randn(random.randint(1, 10), random.randint(1, 10), random.randint(1, 10), random.randint(1, 10))
        fractional_max_pool = torch.nn.FractionalMaxPool2d(kernel_size, output_size=output_size)
        result = fractional_max_pool(input_tensor)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_fractionalmaxpool2d_large_scale(self):
        kernel_size = (random.randint(100, 1000), random.randint(100, 1000))
        output_size = (random.randint(100, 1000), random.randint(100, 1000))
        input_tensor = torch.randn(random.randint(1000, 10000), random.randint(100, 1000), random.randint(100, 1000), random.randint(100, 1000))
        fractional_max_pool = torch.nn.FractionalMaxPool2d(kernel_size, output_size=output_size)
        result = fractional_max_pool(input_tensor)
        return result

