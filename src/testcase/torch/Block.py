
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Block)
class TorchBlockTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_block_correctness(self):
        result = torch.Block()
        return result

    @test_api_version.larger_than("1.1.3")
    def test_block_large_scale(self):
        result = torch.Block()
        return result

