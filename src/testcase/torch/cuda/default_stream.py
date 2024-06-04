
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.default_stream)
class TorchCudaDefaultStreamTestCase(TorBencherTestCaseBase):
    def test_default_stream(self):
        a = torch.device('cuda')
        result = torch.cuda.default_stream(a)
        return result

