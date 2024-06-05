
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.autograd.no_grad)
class TorchAutogradNoGradTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_no_grad_correctness(self):
        result = torch.autograd.no_grad()
        return result

    @test_api_version.larger_than("1.1.3")
    def test_no_grad_large_scale(self):
        result = torch.autograd.no_grad()
        return result


