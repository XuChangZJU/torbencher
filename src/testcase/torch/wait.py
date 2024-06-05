
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.wait)
class TorchWaitTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_wait_correctness(self):
        dim = random.randint(1, 10)
        tensor = torch.randn(dim)
        result = torch.wait(tensor)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_wait_large_scale(self):
        dim = random.randint(1000, 10000)
        tensor = torch.randn(dim)
        result = torch.wait(tensor)
        return result

