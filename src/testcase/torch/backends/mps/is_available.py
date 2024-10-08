import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.backends.mps.is_available)
class TorchBackendsMpsIsUavailableTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_is_available_correctness(self):
        result = torch.backends.mps.is_available()
        return result
