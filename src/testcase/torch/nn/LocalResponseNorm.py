import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.LocalResponseNorm)
class TorchNNLocalResponseNormTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_local_response_norm(self, input=None):
        if input is not None:
            result = torch.nn.LocalResponseNorm(input[0])(input[1])
            return [result, input]
        a = torch.randn(10, 20, 50, 50)
        lrn = torch.nn.LocalResponseNorm(2)
        result = lrn(a)
        return [result, [2, a]]

