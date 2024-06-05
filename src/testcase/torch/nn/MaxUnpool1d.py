
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.MaxUnpool1d)
class TorchMaxUnpool1dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_maxunpool1d_correctness(self):
        kernel_size = random.randint(1, 10)
        stride = random.randint(1, kernel_size)
        padding = random.randint(0, kernel_size)
        input_tensor = torch.randn(random.randint(1, 10), random.randint(1, 10), random.randint(1, 10))
        indices = torch.randint(0, input_tensor.size(-1), (random.randint(1, 10), random.randint(1, 10), random.randint(1, 10)))
        max_unpool = torch.nn.MaxUnpool1d(kernel_size, stride=stride, padding=padding)
        result = max_unpool(input_tensor, indices)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_maxunpool1d_large_scale(self):
        kernel_size = random.randint(100, 1000)
        stride = random.randint(10, kernel_size)
        padding = random.randint(0, kernel_size)
        input_tensor = torch.randn(random.randint(1000, 10000), random.randint(100, 1000), random.randint(100, 1000))
        indices = torch.randint(0, input_tensor.size(-1), (random.randint(1000, 10000), random.randint(100, 1000), random.randint(100, 1000)))
        max_unpool = torch.nn.MaxUnpool1d(kernel_size, stride=stride, padding=padding)
        result = max_unpool(input_tensor, indices)
        return result

