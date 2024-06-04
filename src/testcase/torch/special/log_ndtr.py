
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.special.log_ndtr)
class TorchSpecialLogNdtrTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.8.0")
    def test_log_ndtr_0d(self):
        a = torch.randn([])
        result = torch.special.log_ndtr(a)
        return result

    @test_api_version.larger_than("1.8.0")
    def test_log_ndtr_1d(self):
        a = torch.randn(5)
        result = torch.special.log_ndtr(a)
        return result

