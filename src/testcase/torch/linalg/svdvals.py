
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.linalg.svdvals)
class TorchLinalgSvdvalsTestCase(TorBencherTestCaseBase):
    def test_svdvals_4d(self, input=None):
        if input is not None:
            result = torch.linalg.svdvals(input[0])
            return [result, input]
        a = torch.randn(2, 2, 3, 3)
        result = torch.linalg.svdvals(a)
        return [result, [a]]

