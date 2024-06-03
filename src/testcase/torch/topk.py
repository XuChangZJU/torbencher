
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.topk)
class TorchTopkTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_topk_4d(self, input=None):
        if input is not None:
            result = torch.topk(input[0], input[1])
            return result
        a = torch.randn(4, 4)
        result = torch.topk(a, 2)
        return result

