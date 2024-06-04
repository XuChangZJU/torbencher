
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.reset_max_memory_cached)
class TorchCudaResetMaxMemoryCachedTestCase(TorBencherTestCaseBase):
    def test_reset_max_memory_cached_0(self):
        a = torch.device('cuda')
        result = torch.cuda.reset_max_memory_cached(a)
        return result
    def test_reset_max_memory_cached_1(self):
        a = torch.device('cuda')
        result = torch.cuda.reset_max_memory_cached(device=a)
        return result

