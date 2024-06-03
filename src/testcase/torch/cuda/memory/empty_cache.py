

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.cuda.memory.empty_cache)
class TorchCudaMemoryEmptyCacheTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.9.0")
    def test_empty_cache(self, input=None):
        if input is not None:
            result = torch.cuda.memory.empty_cache()
            return result
        result = torch.cuda.memory.empty_cache()
        return result
