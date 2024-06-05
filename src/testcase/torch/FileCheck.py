
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.FileCheck)
class TorchFileCheckTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_filecheck_correctness(self):
        result = torch.FileCheck()
        return result

    @test_api_version.larger_than("1.1.3")
    def test_filecheck_large_scale(self):
        result = torch.FileCheck()
        return result

