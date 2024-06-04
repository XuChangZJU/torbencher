
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cholesky)
class TorchCholeskyTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_cholesky_2d(self):
        
        a = torch.randn(3, 3)
        a = torch.mm(a, a.t()) + 1e-05 * torch.eye(3) # make symmetric positive-definite
        result = torch.cholesky(a)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_cholesky_2d_upper(self):
        
        a = torch.randn(3, 3)
        a = torch.mm(a, a.t()) + 1e-05 * torch.eye(3) # make symmetric positive-definite
        upper = True
        result = torch.cholesky(a, upper=upper)
        return result

# torch.cholesky_inverse
