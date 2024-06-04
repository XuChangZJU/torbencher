
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.clip)
class TorchClipTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_clip(self):
        a = torch.randn(4)
        result = torch.clip(a, min=-0.5, max=0.5)
        return result


