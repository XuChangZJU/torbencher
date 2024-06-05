
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.layout)
class TorchLayoutTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_layout_correctness(self):
        result = torch.layout.strided
        return result

    @test_api_version.larger_than("1.1.3")
    def test_layout_large_scale(self):
        result = torch.layout.strided
        return result

