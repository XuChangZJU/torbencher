
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.Any)
class TorchCudaAnyTestCase(TorBencherTestCaseBase):
    def test_any_correctness(self):
        result = torch.cuda.Any
        return result

    def test_any_large_scale(self):
        result = torch.cuda.Any
        return result

