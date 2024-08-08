import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.backends.mps.is_built)
class TorchBackendsMpsIsUbuiltTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_is_built_correctness(self):
        # No input parameters needed for torch.backends.mps.is_built
        result = torch.backends.mps.is_built()
        return result
