
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.get_device_name)
class TorchCudaGetDeviceNameTestCase(TorBencherTestCaseBase):
    def test_get_device_name_0(self, input=None):
        if input is not None:
            result = torch.cuda.get_device_name(input[0])
            return result
        a = 0
        result = torch.cuda.get_device_name(a)
        return result

    def test_get_device_name_1(self, input=None):
        if input is not None:
            result = torch.cuda.get_device_name()
            return result
        result = torch.cuda.get_device_name()
        return result
        
