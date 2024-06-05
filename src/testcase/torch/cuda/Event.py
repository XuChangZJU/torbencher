
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.Event)
class TorchCudaEventTestCase(TorBencherTestCaseBase):
    def test_event_correctness(self):
        event = torch.cuda.Event()
        result = event.record()
        return result

    def test_event_large_scale(self):
        event = torch.cuda.Event()
        result = event.record()
        return result

