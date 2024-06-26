
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.group_norm)
class TorchNNFunctionalGroupNormTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_group_norm_common(self, input=None):
        if input is not None:
            result = torch.nn.functional.group_norm(input[0], input[1], weight=input[2], bias=input[3], eps=input[4])
            return [result, input]
        a = torch.randn(20, 5, 10, 10)
        b = 5
        c = None
        d = None
        e = 1e-05
        result = torch.nn.functional.group_norm(a, b, weight=c, bias=d, eps=e)
        return [result, [a, b, c, d, e]]


