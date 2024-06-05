
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.FloatType)
class TorchFloatTypeTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_floattype_correctness(self):
        result = torch.FloatType()
        return result

    @test_api_version.larger_than("1.1.3")
    def test_floattype_large_scale(self):
        result = torch.FloatType()
        return result

