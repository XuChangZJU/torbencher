
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.CELU)
class TorchNNCELUTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_celu(self):
        a = torch.randn(10)
        celu = torch.nn.CELU()
        result = celu(a)
        return result

