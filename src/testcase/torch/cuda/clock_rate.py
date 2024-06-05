
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.clock_rate)
class TorchCudaClockRateTestCase(TorBencherTestCaseBase):
    def test_clock_rate_correctness(self):
        device = random.randint(0, torch.cuda.device_count() - 1)
        result = torch.cuda.clock_rate(device)
        return result

    def test_clock_rate_large_scale(self):
        device = random.randint(0, torch.cuda.device_count() - 1)
        result = torch.cuda.clock_rate(device)
        return result

