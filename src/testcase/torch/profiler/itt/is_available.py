import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.profiler.itt.is_available)
class TorchProfilerIttIsUavailableTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_is_available_correctness(self):
        # No specific input parameters needed for is_available()
        result_is_available = torch.profiler.itt.is_available()
        return result_is_available
