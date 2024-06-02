
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.atleast_1d)
class TorchAtleast1dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_atleast_1d(self, input=None):
        if input is not None:
            result = torch.atleast_1d(input[0])
            return [result, input]
        a = torch.randn(4)
        result = torch.atleast_1d(a)
        return [result, [a]]

    @test_api_version.larger_than("1.1.3")
    def test_atleast_1d_scalar(self, input=None):
        if input is not None:
            result = torch.atleast_1d(input[0])
            return [result, input]
        a = torch.tensor(1.2)
        result = torch.atleast_1d(a)
        return [result, [a]]


