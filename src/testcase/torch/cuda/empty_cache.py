
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.empty_cache)
class TorchCudaEmptyCacheTestCase(TorBencherTestCaseBase):
    def test_empty_cache(self):
        
        result = torch.cuda.empty_cache()
        return result

