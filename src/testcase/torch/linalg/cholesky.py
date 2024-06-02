
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.linalg.cholesky)
class TorchLinalgCholeskyTestCase(TorBencherTestCaseBase):
    def test_cholesky_4d(self, input=None):
        if input is not None:
            result = torch.linalg.cholesky(input[0])
            return [result, input]
        a = torch.randn(2, 2, 2, 2)
        a = torch.matmul(a, a.transpose(-1, -2)) + 1e-05 * torch.eye(2, 2)
        result = torch.linalg.cholesky(a)
        return [result, [a]]

