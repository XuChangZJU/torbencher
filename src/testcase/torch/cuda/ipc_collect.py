
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.ipc_collect)
class TorchCudaIpcCollectTestCase(TorBencherTestCaseBase):
    def test_ipc_collect(self, input=None):
        if input is not None:
            result = torch.cuda.ipc_collect()
            return [result, input]
        result = torch.cuda.ipc_collect()
        return [result, None]

