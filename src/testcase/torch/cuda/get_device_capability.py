
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.get_device_capability)
class TorchCudaGetDeviceCapabilityTestCase(TorBencherTestCaseBase):
    def test_get_device_capability_0(self):
        a = 0
        result = torch.cuda.get_device_capability(a)
        return result
    def test_get_device_capability_1(self):
        a = 0
        result = torch.cuda.get_device_capability(device=a)
        return result

