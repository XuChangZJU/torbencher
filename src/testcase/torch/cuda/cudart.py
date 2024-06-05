
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.cudart)
class TorchCudaCudartTestCase(TorBencherTestCaseBase):
    def test_cudart_correctness(self):
        result = torch.cuda.cudart
        return result

    def test_cudart_large_scale(self):
        result = torch.cuda.cudart
        return result





