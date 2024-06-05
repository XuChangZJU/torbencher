
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.memory.Any)
class AnyTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_any_correctness(self):
        result = torch.cuda.memory.Any()
        return result

    @test_api_version.larger_than("1.1.3")
    def test_any_large_scale(self):
        result = torch.cuda.memory.Any()
        return result

