
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.atleast_3d)
class TorchAtleast3dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_atleast_3d(self, input=None):
        if input is not None:
            result = torch.atleast_3d(input[0])
            return result
        a = torch.randn(4)
        result = torch.atleast_3d(a)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_atleast_3d_scalar(self, input=None):
        if input is not None:
            result = torch.atleast_3d(input[0])
            return result
        a = torch.tensor(1.2)
        result = torch.atleast_3d(a)
        return result


