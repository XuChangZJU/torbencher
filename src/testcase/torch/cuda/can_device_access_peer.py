
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.can_device_access_peer)
class TorchCudaCanDeviceAccessPeerTestCase(TorBencherTestCaseBase):
    def test_can_device_access_peer_correctness(self):
        device1 = random.randint(0, torch.cuda.device_count() - 1)
        device2 = random.randint(0, torch.cuda.device_count() - 1)
        result = torch.cuda.can_device_access_peer(device1, device2)
        return result

    def test_can_device_access_peer_large_scale(self):
        device1 = random.randint(0, torch.cuda.device_count() - 1)
        device2 = random.randint(0, torch.cuda.device_count() - 1)
        result = torch.cuda.can_device_access_peer(device1, device2)
        return result

