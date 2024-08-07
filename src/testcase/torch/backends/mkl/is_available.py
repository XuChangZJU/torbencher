import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.backends.mkl.is_available)
class TorchBackendsMklIsUavailableTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_is_available_correctness(self):
        # No input parameters needed for torch.backends.mkl.is_available
        result_is_available = torch.backends.mkl.is_available()
        return result_is_available
