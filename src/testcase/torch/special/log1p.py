import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.special.log1p)
class TorchSpecialLog1pTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_special_log1p_4d(self, input=None):
        if input is not None:
            result = torch.special.log1p(input[0])
            return [result, input]
        a = torch.randn(4)
        result = torch.special.log1p(a)
        return [result, [a]]

