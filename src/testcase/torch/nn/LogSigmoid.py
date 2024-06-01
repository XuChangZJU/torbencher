import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.LogSigmoid)
class TorchNNLogSigmoidTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_log_sigmoid(self, input=None):
        if input is not None:
            result = torch.nn.LogSigmoid()(input[0])
            return [result, input]
        a = torch.randn(10)
        log_sigmoid = torch.nn.LogSigmoid()
        result = log_sigmoid(a)
        return [result, [a]]

