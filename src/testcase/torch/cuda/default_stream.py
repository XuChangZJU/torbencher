
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.default_stream)
class TorchCudaDefaultStreamTestCase(TorBencherTestCaseBase):
    def test_default_stream(self, input=None):
        if input is not None:
            result = torch.cuda.default_stream(input[0])
            return [result, input]
        a = torch.device('cuda')
        result = torch.cuda.default_stream(a)
        return [result, [a]]

