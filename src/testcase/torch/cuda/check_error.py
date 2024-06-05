
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.check_error)
class TorchCudaCheckErrorTestCase(TorBencherTestCaseBase):
    def test_check_error_correctness(self):
        result = torch.cuda.check_error()
        return result

    def test_check_error_large_scale(self):
        result = torch.cuda.check_error()
        return result

