import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.profiler.itt.range_push)
class TorchProfilerIttRangeUpushTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_range_push_correctness(self):
        # No random parameters for torch.profiler.itt.range_push
        msg = "test_message"  # type: str
        result = torch.profiler.itt.range_push(msg)
        return result
