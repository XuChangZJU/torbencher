
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.addcmul)
class TorchAddcmulTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_addcmul(self):
        a = torch.randn(4)
        b = torch.randn(4)
        c = torch.randn(4)
        result = torch.addcmul(a, b, c, value=10)
        return result


