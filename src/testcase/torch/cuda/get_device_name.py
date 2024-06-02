
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.get_device_name)
class TorchCudaGetDeviceNameTestCase(TorBencherTestCaseBase):
    def test_get_device_name_0(self, input=None):
        if input is not None:
            result = torch.cuda.get_device_name(input[0])
            return [result, input]
        a = 0
        result = torch.cuda.get_device_name(a)
        return [result, [a]]

    def test_get_device_name_1(self, input=None):
        if input is not None:
            result = torch.cuda.get_device_name()
            return [result, input]
        result = torch.cuda.get_device_name()
        return [result, None]
        
