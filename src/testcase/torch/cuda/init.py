
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.init)
class TorchCudaInitTestCase(TorBencherTestCaseBase):
    def test_init_correctness(self):
        result = torch.cuda.init()
        return result

    def test_init_large_scale(self):
        result = torch.cuda.init()
        return result

