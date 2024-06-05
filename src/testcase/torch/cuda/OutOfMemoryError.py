
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.OutOfMemoryError)
class TorchCudaOutOfMemoryErrorTestCase(TorBencherTestCaseBase):
    def test_outofmemoryerror_correctness(self):
        error = torch.cuda.OutOfMemoryError()
        result = error.type
        return result

    def test_outofmemoryerror_large_scale(self):
        error = torch.cuda.OutOfMemoryError()
        result = error.type
        return result

