
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.device_count)
class TorchCudaDeviceCountTestCase(TorBencherTestCaseBase):
    def test_device_count(self):
        
        result = torch.cuda.device_count()
        return result

