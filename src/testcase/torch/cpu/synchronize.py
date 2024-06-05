
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cpu.synchronize)
class TorchCpuSynchronizeTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_synchronize_correctness(self):
        torch.cuda.synchronize()
        return None

    @test_api_version.larger_than("1.1.3")
    def test_synchronize_large_scale(self):
        torch.cuda.synchronize()
        return None



