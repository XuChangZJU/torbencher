import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.autograd.set_detect_anomaly)
class TorchAutogradSetdetectanomalyTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_set_detect_anomaly_correctness(self):
        mode = random.choice([True, False])  # Randomly choose mode
        check_nan = random.choice([True, False]) # Randomly choose check_nan
        result = torch.autograd.set_detect_anomaly(mode, check_nan)
        return result
    