
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.autograd.kineto_available)
class TorchAutogradKinetoAvailableTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_kineto_available_correctness(self):
        result = torch.autograd.kineto_available()
        return result

    @test_api_version.larger_than("1.1.3")
    def test_kineto_available_large_scale(self):
        result = torch.autograd.kineto_available()
        return result


