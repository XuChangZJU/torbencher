
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.is_available)
class TorchCudaIsAvailableTestCase(TorBencherTestCaseBase):
    def test_is_available_correctness(self):
        result = torch.cuda.is_available()
        return result

    def test_is_available_large_scale(self):
        result = torch.cuda.is_available()
        return result

