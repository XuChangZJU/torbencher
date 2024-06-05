
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.default_stream)
class TorchCudaDefaultStreamTestCase(TorBencherTestCaseBase):
    def test_default_stream_correctness(self):
        device = random.randint(0, torch.cuda.device_count() - 1)
        result = torch.cuda.default_stream(device)
        return result

    def test_default_stream_large_scale(self):
        device = random.randint(0, torch.cuda.device_count() - 1)
        result = torch.cuda.default_stream(device)
        return result

