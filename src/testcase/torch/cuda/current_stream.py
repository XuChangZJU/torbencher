
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.current_stream)
class TorchCudaCurrentStreamTestCase(TorBencherTestCaseBase):
    def test_current_stream(self):
        a = torch.device('cuda')
        result = torch.cuda.current_stream(a)
        return result

