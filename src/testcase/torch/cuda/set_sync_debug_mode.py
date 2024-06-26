
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.set_sync_debug_mode)
class TorchCudaSetSyncDebugModeTestCase(TorBencherTestCaseBase):
    def test_set_sync_debug_mode_0(self, input=None):
        if input is not None:
            result = torch.cuda.set_sync_debug_mode(input[0])
            return [result, input]
        a = 0
        result = torch.cuda.set_sync_debug_mode(a)
        return [result, [a]]
    def test_set_sync_debug_mode_1(self, input=None):
        if input is not None:
            result = torch.cuda.set_sync_debug_mode(mode=input[0])
            return [result, input]
        a = 0
        result = torch.cuda.set_sync_debug_mode(mode=a)
        return [result, [a]]

