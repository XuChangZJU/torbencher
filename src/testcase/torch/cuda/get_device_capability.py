
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.get_device_capability)
class TorchCudaGetDeviceCapabilityTestCase(TorBencherTestCaseBase):
    def test_get_device_capability_0(self, input=None):
        if input is not None:
            result = torch.cuda.get_device_capability(input[0])
            return [result, input]
        a = 0
        result = torch.cuda.get_device_capability(a)
        return [result, [a]]
    def test_get_device_capability_1(self, input=None):
        if input is not None:
            result = torch.cuda.get_device_capability(device=input[0])
            return [result, input]
        a = 0
        result = torch.cuda.get_device_capability(device=a)
        return [result, [a]]

