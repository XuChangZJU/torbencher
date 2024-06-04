
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.linalg.cholesky_ex)
class TorchLinalgCholeskyExTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.8.2")
    def test_cholesky_ex(self):
        a = torch.rand(3, 3)
        a = torch.mm(a, a.t())
        result = torch.linalg.cholesky_ex(a)
        return result

