
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.amp.autocast)
class AutocastTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.6.0")
    def test_autocast_type(self):
        autocast = torch.cuda.amp.autocast()
        result = autocast.type
        return result

    @test_api_version.larger_than("1.6.0")
    def test_autocast_type_large_scale(self):
        autocast = torch.cuda.amp.autocast()
        result = autocast.type
        return result

