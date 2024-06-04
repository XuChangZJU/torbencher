
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.linalg.ldl_factor)
class TorchLinalgLdlFactorTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.10.0")
    def test_ldl_factor(self):
        
        a = torch.randn(3, 3)
        a = (a + a.t()) / 2  # make symmetric
        result = torch.linalg.ldl_factor(a, hermitian=True)
        return result

