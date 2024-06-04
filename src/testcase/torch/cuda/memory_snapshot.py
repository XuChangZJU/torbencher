
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.memory_snapshot)
class TorchCudaMemorySnapshotTestCase(TorBencherTestCaseBase):
    def test_memory_snapshot(self):
        
        result = torch.cuda.memory_snapshot()
        return result

