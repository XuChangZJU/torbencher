
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.empty_cache)
class TorchCudaEmptyCacheTestCase(TorBencherTestCaseBase):
    def test_empty_cache_correctness(self):
        result = torch.cuda.empty_cache()
        return result

    def test_empty_cache_large_scale(self):
        result = torch.cuda.empty_cache()
        return result

