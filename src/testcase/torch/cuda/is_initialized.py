
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.is_initialized)
class TorchCudaIsInitializedTestCase(TorBencherTestCaseBase):
    def test_is_initialized_correctness(self):
        result = torch.cuda.is_initialized()
        return result

    def test_is_initialized_large_scale(self):
        result = torch.cuda.is_initialized()
        return result

