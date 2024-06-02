
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.init)
class TorchCudaInitTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_init(self, input=None):
        if input is not None:
            result = torch.cuda.init()
            return [result, input]
        result = torch.cuda.init()
        return [result, None]


