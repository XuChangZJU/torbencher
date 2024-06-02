
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.addcmul)
class TorchAddcmulTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_addcmul(self, input=None):
        if input is not None:
            result = torch.addcmul(input[0], input[1], input[2], value=input[3])
            return [result, input]
        a = torch.randn(4)
        b = torch.randn(4)
        c = torch.randn(4)
        result = torch.addcmul(a, b, c, value=10)
        return [result, [a, b, c, 10]]


