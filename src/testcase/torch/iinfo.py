
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.iinfo)
class TorchIinfoTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_iinfo_correctness(self):
        result = torch.iinfo(torch.int32)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_iinfo_large_scale(self):
        result = torch.iinfo(torch.int32)
        return result

