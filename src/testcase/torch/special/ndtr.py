
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.special.ndtr)
class TorchSpecialNdtrTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_ndtr_0d(self, input=None):
        if input is not None:
            result = torch.special.ndtr(input[0])
            return result
        a = torch.randn([])
        result = torch.special.ndtr(a)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_ndtr_1d(self, input=None):
        if input is not None:
            result = torch.special.ndtr(input[0])
            return result
        a = torch.randn(5)
        result = torch.special.ndtr(a)
        return result

