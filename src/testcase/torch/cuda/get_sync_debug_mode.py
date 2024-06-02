
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.get_sync_debug_mode)
class TorchCudaGetSyncDebugModeTestCase(TorBencherTestCaseBase):
    def test_get_sync_debug_mode(self, input=None):
        if input is not None:
            result = torch.cuda.get_sync_debug_mode()
            return [result, input]
        result = torch.cuda.get_sync_debug_mode()
        return [result, None]

