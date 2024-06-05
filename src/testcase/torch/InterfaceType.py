
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.InterfaceType)
class TorchInterfaceTypeTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_interfacetype_correctness(self):
        result = torch.InterfaceType()
        return result

    @test_api_version.larger_than("1.1.3")
    def test_interfacetype_large_scale(self):
        result = torch.InterfaceType()
        return result

