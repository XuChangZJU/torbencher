
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.set_device)
class TorchCudaSetDeviceTestCase(TorBencherTestCaseBase):
    def test_set_device_0(self):
        a = 0
        torch.cuda.set_device(a)
        result = torch.cuda.current_device()
        return result
    def test_set_device_1(self):
        torch.cuda.set_device('cuda:0')
        result = torch.cuda.current_device()
        return result
