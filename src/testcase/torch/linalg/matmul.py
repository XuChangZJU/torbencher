
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.linalg.matmul)
class TorchLinalgMatmulTestCase(TorBencherTestCaseBase):
    def test_matmul_4d(self):
        
        a = torch.randn(2, 2, 3, 4)
        b = torch.randn(2, 2, 4, 3)
        result = torch.linalg.matmul(a, b)
        return result

