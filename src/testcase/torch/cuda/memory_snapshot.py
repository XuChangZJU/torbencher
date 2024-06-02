
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.memory_snapshot)
class TorchCudaMemorySnapshotTestCase(TorBencherTestCaseBase):
    def test_memory_snapshot(self, input=None):
        if input is not None:
            result = torch.cuda.memory_snapshot()
            return [result, input]
        result = torch.cuda.memory_snapshot()
        return [result, None]

