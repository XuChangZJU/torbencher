
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.cudart)
class TorchCudaCudartTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_cudart(self, input=None):
        if input is not None:
            result = torch.cuda.cudart()
            return [result, input]
        result = torch.cuda.cudart()
        return [result, None]




