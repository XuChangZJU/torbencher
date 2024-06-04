
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cholesky_inverse)
class TorchCholeskyInverseTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_cholesky_inverse_2d(self):
        
        a = torch.randn(3, 3)
        a = torch.mm(a, a.t()) + 1e-05 * torch.eye(3) # make symmetric positive-definite
        l = torch.cholesky(a)
        result = torch.cholesky_inverse(l)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_cholesky_inverse_2d_upper(self):
        
        a = torch.randn(3, 3)
        a = torch.mm(a, a.t()) + 1e-05 * torch.eye(3) # make symmetric positive-definite
        l = torch.cholesky(a)
        upper = True
        result = torch.cholesky_inverse(l, upper=upper)
        return result

