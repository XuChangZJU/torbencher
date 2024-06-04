
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.full)
class TorchFullTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_full(self):
        result = torch.full((2, 3), 3.141592)
        return result

