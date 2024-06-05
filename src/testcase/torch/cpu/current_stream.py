
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cpu.current_stream)
class TorchCpuCurrentStreamTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_current_stream_correctness(self):
        result = torch.cuda.current_stream()
        return result

    @test_api_version.larger_than("1.1.3")
    def test_current_stream_large_scale(self):
        result = torch.cuda.current_stream()
        return result


