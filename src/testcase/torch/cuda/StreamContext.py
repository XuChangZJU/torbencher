
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.StreamContext)
class TorchCudaStreamContextTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.7.0")
    def test_streamcontext_correctness(self):
        context = torch.cuda.StreamContext()
        result = context.type
        return result

    @test_api_version.larger_than("1.7.0")
    def test_streamcontext_large_scale(self):
        context = torch.cuda.StreamContext()
        result = context.type
        return result

