
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.special.multigammaln)
class TorchSpecialMultigammalnTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_multigammaln_0d(self):
        
        a = torch.rand([])
        b = torch.randint(1, 5, ())
        result = torch.special.multigammaln(a, b)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_multigammaln_1d(self):
        
        a = torch.rand(5)
        b = torch.randint(1, 5, ())
        result = torch.special.multigammaln(a, b)
        return result

