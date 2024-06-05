
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.kl_div)
class KLDivTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_kl_div_correctness(self):
        input_data = torch.randn(10, 10)
        target = torch.randn(10, 10)
        log_target = random.choice([True, False])
        reduction = random.choice(['mean', 'sum', 'none'])
        result = torch.nn.functional.kl_div(input_data, target, log_target, reduction)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_kl_div_large_scale(self):
        input_data = torch.randn(1000, 1000)
        target = torch.randn(1000, 1000)
        log_target = random.choice([True, False])
        reduction = random.choice(['mean', 'sum', 'none'])
        result = torch.nn.functional.kl_div(input_data, target, log_target, reduction)
        return result

