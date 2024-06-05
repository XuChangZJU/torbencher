
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Future)
class TorchFutureTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_future_correctness(self):
        result = torch.Future()
        return result

    @test_api_version.larger_than("1.1.3")
    def test_future_large_scale(self):
        result = torch.Future()
        return result

