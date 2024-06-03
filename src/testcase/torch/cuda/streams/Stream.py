
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.cuda.streams.Stream)
class TorchCudaStreamsStreamTestCase(TorBencherTestCaseBase):
    def test_stream(self, input=None):
        if input is not None:
            result = torch.cuda.streams.Stream(input[0], input[1])
            return result
        a = torch.device('cuda')
        b = 0
        result = torch.cuda.streams.Stream(a, b)
        return result