
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.set_stream)
class TorchCudaSetStreamTestCase(TorBencherTestCaseBase):
    def test_set_stream_correctness(self):
        stream = torch.cuda.Stream()
        result = torch.cuda.set_stream(stream)
        return result

    def test_set_stream_large_scale(self):
        stream = torch.cuda.Stream()
        result = torch.cuda.set_stream(stream)
        return result

