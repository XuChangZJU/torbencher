import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.Sigmoid)
class TorchNNSigmoidTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_sigmoid(self, input=None):
        if input is not None:
            result = torch.nn.Sigmoid()(input[0])
            return [result, input]
        a = torch.randn(10)
        sigmoid = torch.nn.Sigmoid()
        result = sigmoid(a)
        return [result, [a]]

