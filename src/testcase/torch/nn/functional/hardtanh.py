
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.hardtanh)
class HardtanhTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_hardtanh_correctness(self):
        input_data = torch.randn(10, 10)
        min_value = random.uniform(-10.0, 10.0)
        max_value = random.uniform(-10.0, 10.0)
        result = torch.nn.functional.hardtanh(input_data, min_value, max_value)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_hardtanh_large_scale(self):
        input_data = torch.randn(1000, 1000)
        min_value = random.uniform(-10.0, 10.0)
        max_value = random.uniform(-10.0, 10.0)
        result = torch.nn.functional.hardtanh(input_data, min_value, max_value)
        return result

