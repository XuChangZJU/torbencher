
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.autograd.detect_anomaly)
class TorchAutogradDetectAnomalyTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_detect_anomaly_correctness(self):
        result = torch.autograd.detect_anomaly()
        return result

    @test_api_version.larger_than("1.1.3")
    def test_detect_anomaly_large_scale(self):
        result = torch.autograd.detect_anomaly()
        return result


