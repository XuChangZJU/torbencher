
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.device)
class TorchCudaDeviceTestCase(TorBencherTestCaseBase):
    def test_device_0(self):
        a = 0
        with torch.cuda.device(a):
            b = torch.tensor([1])
            result = b.to('cuda:0')
        return result
    def test_device_1(self):
        with torch.cuda.device('cuda:0'):
            b = torch.tensor([1])
            result = b.to('cuda:0')
        return result

