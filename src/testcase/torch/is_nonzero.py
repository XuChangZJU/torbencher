
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.is_nonzero)
class TorchIsNonzeroTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_is_nonzero_correctness(self):
        tensor = torch.randn(1)
        result = torch.is_nonzero(tensor)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_is_nonzero_large_scale(self):
        tensor = torch.randn(1)
        result = torch.is_nonzero(tensor)
        return result

