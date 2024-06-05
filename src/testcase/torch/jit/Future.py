
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.jit.Future)
class TorchJitFutureTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_future_correctness(self):
        future = torch.jit.Future()
        result = future.add_done_callback(lambda _: None)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_future_large_scale(self):
        future = torch.jit.Future()
        result = future.add_done_callback(lambda _: None)
        return result

