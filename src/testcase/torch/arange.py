
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.arange)
class TorchArnageTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_arange_correctness(self):
        end = random.randint(1, 10)
        result = torch.arange(end)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_arange_large_scale(self):
        end = random.randint(1000, 10000)
        result = torch.arange(end)
        return result

