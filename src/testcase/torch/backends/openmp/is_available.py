import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.backends.openmp.is_available)
class TorchBackendsOpenmpIsUavailableTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_is_available_correctness(self):
        # No specific input parameters needed for torch.backends.openmp.is_available
        result_is_available = torch.backends.openmp.is_available()  # Check if OpenMP is available
        return result_is_available
