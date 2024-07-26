import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.backends.mps.is_built)
class TorchBackendsMpsIsbuiltTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_is_built_correctness(self):
        # No input parameters needed for torch.backends.mps.is_built
        result = torch.backends.mps.is_built()
        return result
