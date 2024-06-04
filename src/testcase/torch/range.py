
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.range)
class TorchRangeTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_range(self):
        result = torch.range(1, 4, 0.5)
        return result

