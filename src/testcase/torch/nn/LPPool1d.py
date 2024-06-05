
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.LPPool1d)
class TorchLPPool1dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_lppool1d_correctness(self):
        norm_type = random.randint(1, 10)
        kernel_size = random.randint(1, 10)
        stride = random.randint(1, kernel_size)
        input_tensor = torch.randn(random.randint(1, 10), random.randint(1, 10), random.randint(1, 10))
        lp_pool = torch.nn.LPPool1d(norm_type, kernel_size, stride=stride)
        result = lp_pool(input_tensor)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_lppool1d_large_scale(self):
        norm_type = random.randint(1, 10)
        kernel_size = random.randint(100, 1000)
        stride = random.randint(10, kernel_size)
        input_tensor = torch.randn(random.randint(1000, 10000), random.randint(100, 1000), random.randint(100, 1000))
        lp_pool = torch.nn.LPPool1d(norm_type, kernel_size, stride=stride)
        result = lp_pool(input_tensor)
        return result

