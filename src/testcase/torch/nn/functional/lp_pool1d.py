
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.lp_pool1d)
class LpPool1dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_lp_pool1d_correctness(self):
        batch_size = random.randint(1, 10)
        channel = random.randint(1, 10)
        length = random.randint(10, 20)
        kernel_size = random.randint(1, 5)
        stride = random.randint(1, 3)
        padding = random.randint(0, 2)
        norm_type = random.randint(1, 5)
        ceil_mode = random.choice([True, False])
        input_data = torch.randn(batch_size, channel, length)
        result = torch.nn.functional.lp_pool1d(input_data, norm_type, kernel_size, stride, padding, ceil_mode)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_lp_pool1d_large_scale(self):
        batch_size = random.randint(100, 1000)
        channel = random.randint(100, 1000)
        length = random.randint(1000, 2000)
        kernel_size = random.randint(10, 50)
        stride = random.randint(1, 10)
        padding = random.randint(0, 10)
        norm_type = random.randint(1, 5)
        ceil_mode = random.choice([True, False])
        input_data = torch.randn(batch_size, channel, length)
        result = torch.nn.functional.lp_pool1d(input_data, norm_type, kernel_size, stride, padding, ceil_mode)
        return result

