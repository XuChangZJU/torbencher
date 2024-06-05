
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.frombuffer)
class TorchFrombufferTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_frombuffer_correctness(self):
        data = bytes(random.randint(1, 10))
        result = torch.frombuffer(data)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_frombuffer_large_scale(self):
        data = bytes(random.randint(1000, 10000))
        result = torch.frombuffer(data)
        return result

