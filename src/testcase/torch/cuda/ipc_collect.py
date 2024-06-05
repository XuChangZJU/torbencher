
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.ipc_collect)
class TorchCudaIpcCollectTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.7.0")
    def test_ipc_collect_correctness(self):
        result = torch.cuda.ipc_collect()
        return result

    @test_api_version.larger_than("1.7.0")
    def test_ipc_collect_large_scale(self):
        result = torch.cuda.ipc_collect()
        return result

