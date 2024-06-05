
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.set_flush_denormal)
class TorchSetFlushDenormalTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_set_flush_denormal_correctness(self):
        result = torch.set_flush_denormal(True)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_set_flush_denormal_large_scale(self):
        result = torch.set_flush_denormal(False)
        return result

