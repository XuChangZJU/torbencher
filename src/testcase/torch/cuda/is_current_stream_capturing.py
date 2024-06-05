
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.is_current_stream_capturing)
class TorchCudaIsCurrentStreamCapturingTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.7.0")
    def test_is_current_stream_capturing_correctness(self):
        result = torch.cuda.is_current_stream_capturing()
        return result

    @test_api_version.larger_than("1.7.0")
    def test_is_current_stream_capturing_large_scale(self):
        result = torch.cuda.is_current_stream_capturing()
        return result

