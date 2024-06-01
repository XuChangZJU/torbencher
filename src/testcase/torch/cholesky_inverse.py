import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cholesky_inverse)
class TorchCholeskyInverseTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_cholesky_inverse_4d(self, input=None):
        if input is not None:
            result = torch.cholesky_inverse(input[0])
            return [result, input]
        a = torch.randn(4, 4)
        a = a @ a.t()  # Make a symmetric positive definite matrix
        l = torch.cholesky(a)
        result = torch.cholesky_inverse(l)
        return [result, [l]]

