
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.special.airy_ai)
class TorchSpecialAiryAiTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.7.0")
    def test_airy_ai_0d(self, input=None):
        if input is not None:
            result = torch.special.airy_ai(input[0])
            return [result, input]
        a = torch.randn([])
        result = torch.special.airy_ai(a)
        return [result, [a]]

    @test_api_version.larger_than("1.7.0")
    def test_airy_ai_1d(self, input=None):
        if input is not None:
            result = torch.special.airy_ai(input[0])
            return [result, input]
        a = torch.randn(5)
        result = torch.special.airy_ai(a)
        return [result, [a]]


