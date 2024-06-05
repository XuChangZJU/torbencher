
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.Threshold)
class TorchThresholdTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_threshold_correctness(self):
        threshold = random.uniform(0.1, 10.0)
        value = random.uniform(0.1, 10.0)
        input_tensor = torch.randn(random.randint(1, 10), random.randint(1, 10))
        threshold_layer = torch.nn.Threshold(threshold=threshold, value=value)
        result = threshold_layer(input_tensor)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_threshold_large_scale(self):
        threshold = random.uniform(0.1, 10.0)
        value = random.uniform(0.1, 10.0)
        input_tensor = torch.randn(random.randint(1000, 10000), random.randint(100, 1000))
        threshold_layer = torch.nn.Threshold(threshold=threshold, value=value)
        result = threshold_layer(input_tensor)
        return result

