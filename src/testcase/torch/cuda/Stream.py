
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.stream)
class TorchCudaStreamTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_stream_0(self):
        a = torch.cuda.Stream()
        result = torch.cuda.stream(a)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_stream_1(self):
        a = torch.cuda.Stream()
        result = torch.cuda.stream(stream=a)
        return result


