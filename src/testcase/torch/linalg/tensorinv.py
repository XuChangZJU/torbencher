
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.linalg.tensorinv)
class TorchLinalgTensorinvTestCase(TorBencherTestCaseBase):
    def test_tensorinv_4d(self, input=None):
        if input is not None:
            result = torch.linalg.tensorinv(input[0], ind=input[1])
            return result
        a = torch.randn(2, 2, 2, 2)
        ind = 2
        result = torch.linalg.tensorinv(a, ind=ind)
        return result

