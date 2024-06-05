
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cpu.Optional)
class TorchCpuOptionalTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_optional_correctness(self):
        # No specific attribute to test, so we simply return None
        return None

    @test_api_version.larger_than("1.1.3")
    def test_optional_large_scale(self):
        # No specific attribute to test, so we simply return None
        return None


