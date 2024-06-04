
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.can_device_access_peer)
class TorchCudaCanDeviceAccessPeerTestCase(TorBencherTestCaseBase):
    def test_can_device_access_peer_0(self):
        
        a = torch.device('cuda')
        b = torch.device('cuda:0')
        result = torch.cuda.can_device_access_peer(a, b)
        return result

