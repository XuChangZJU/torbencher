
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.is_initialized)
class TorchCudaIsInitializedTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_is_initialized(self, input=None):
        if input is not None:
            result = torch.cuda.is_initialized()
            return result
        result = torch.cuda.is_initialized()
        return result


