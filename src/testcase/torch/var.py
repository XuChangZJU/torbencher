
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.var)
class TorchVarTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_var_dim(self):
        a = torch.randn(4, 4)
        result = torch.var(a, 1)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_var(self):
        a = torch.randn(4, 4)
        result = torch.var(a)
        return result

