
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.hub.URLError)
class TorchHubURLErrorTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_add_note_correctness(self):
        error = torch.hub.URLError(reason="Connection refused")
        error.add_note("Test note")
        result = error.reason
        return result

    @test_api_version.larger_than("1.1.3")
    def test_add_note_large_scale(self):
        error = torch.hub.URLError(reason="Connection timeout")
        error.add_note("This is a large scale test note")
        error.add_note("Another note")
        result = error.reason
        return result

