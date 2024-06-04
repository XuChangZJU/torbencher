
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.manual_seed)
class TorchCudaManualSeedTestCase(TorBencherTestCaseBase):
    def test_manual_seed(self):
        a = 10
        result = torch.cuda.manual_seed(a)
        return result

