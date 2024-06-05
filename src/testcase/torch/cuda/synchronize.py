
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.synchronize)
class TorchCudaSynchronizeTestCase(TorBencherTestCaseBase):
    def test_synchronize_correctness(self):
        result = torch.cuda.synchronize()
        return result

    def test_synchronize_large_scale(self):
        result = torch.cuda.synchronize()
        return result

