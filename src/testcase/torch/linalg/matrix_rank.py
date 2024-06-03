
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.linalg.matrix_rank)
class TorchLinalgMatrixRankTestCase(TorBencherTestCaseBase):
    def test_matrix_rank_4d(self, input=None):
        if input is not None:
            result = torch.linalg.matrix_rank(input[0])
            return result
        a = torch.randn(2, 2, 3, 3)
        result = torch.linalg.matrix_rank(a)
        return result

