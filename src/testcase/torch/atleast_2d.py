
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.atleast_2d)
class TorchAtleast2dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_atleast_2d(self):
        a = torch.randn(4)
        result = torch.atleast_2d(a)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_atleast_2d_scalar(self):
        a = torch.tensor(1.2)
        result = torch.atleast_2d(a)
        return result


