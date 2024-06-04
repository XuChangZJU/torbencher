
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.svd)
class TorchSvdTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_svd_4d(self):
        a = torch.randn(4, 4)
        result = torch.svd(a)
        return result

