
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.Stream)
class TorchCudaStreamTestCase(TorBencherTestCaseBase):
    def test_stream_correctness(self):
        stream = torch.cuda.Stream()
        result = stream.priority_range
        return result

    def test_stream_large_scale(self):
        stream = torch.cuda.Stream()
        result = stream.priority_range
        return result

