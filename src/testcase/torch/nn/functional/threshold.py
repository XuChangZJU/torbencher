
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.threshold)
class ThresholdTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_threshold_correctness(self):
        input_data = torch.randn(10, 10)
        threshold = random.uniform(-10.0, 10.0)
        value = random.uniform(-10.0, 10.0)
        result = torch.nn.functional.threshold(input_data, threshold, value)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_threshold_large_scale(self):
        input_data = torch.randn(1000, 1000)
        threshold = random.uniform(-10.0, 10.0)
        value = random.uniform(-10.0, 10.0)
        result = torch.nn.functional.threshold(input_data, threshold, value)
        return result

