
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tag)
class TorchTagTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_tag_correctness(self):
        result = torch.Tag("test_tag")
        return result

    @test_api_version.larger_than("1.1.3")
    def test_tag_large_scale(self):
        result = torch.Tag("test_tag")
        return result

